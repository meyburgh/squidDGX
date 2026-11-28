// Weight packing, DSpark fold, and CUDA Graph utility kernels for squidDGX.
// Directly callable from C++20 host code inside namespace squid (zero C-ABI FFI).

#include "config.hpp"
#include "kernels.hpp"
#include "kernels_common.cuh"
#include "weights.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace squid {

namespace {

__global__ void bf16_amax_kernel(const __nv_bfloat16* __restrict__ x, size_t n, float* __restrict__ out_amax) {
    float local_max = 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        local_max = fmaxf(local_max, fabsf(__bfloat162float(x[i])));
    }
    local_max = warp_max(local_max);
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        v = warp_max(v);
        if (lane == 0) atomicMax(reinterpret_cast<int*>(out_amax), __float_as_int(v));
    }
}

__device__ __forceinline__ uint8_t quantize_e2m1(float v) {
    uint8_t sign = (v < 0.0f) ? 8u : 0u;
    float a = fabsf(v);
    uint8_t code;
    if      (a < 0.25f) code = 0;
    else if (a < 0.75f) code = 1;
    else if (a < 1.25f) code = 2;
    else if (a < 1.75f) code = 3;
    else if (a < 2.50f) code = 4;
    else if (a < 3.50f) code = 5;
    else if (a < 5.00f) code = 6;
    else                code = 7;
    return sign | code;
}

__global__ void quant_bf16_to_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ src, uint8_t* __restrict__ dst_w, uint8_t* __restrict__ dst_s,
    int N, int K, float global_scale) {
    size_t total_blocks = (size_t)N * (K / 16);
    size_t bidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bidx >= total_blocks) return;
    const __nv_bfloat16* row_in = src + bidx * 16;
    float vals[16];
    float bmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] = __bfloat162float(row_in[i]);
        bmax = fmaxf(bmax, fabsf(vals[i]));
    }
    float raw_scale = (global_scale > 0.0f) ? fminf(bmax / (6.0f * global_scale), 448.0f) : 0.0f;
    __nv_fp8_e4m3 s_fp8(raw_scale);
    dst_s[bidx] = s_fp8.__x;
    float s_eff = static_cast<float>(s_fp8) * global_scale;
    float inv_s = (s_eff > 1e-30f) ? (1.0f / s_eff) : 0.0f;
    uint8_t* out_w = dst_w + bidx * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_w[i] = quantize_e2m1(vals[2 * i] * inv_s) | (quantize_e2m1(vals[2 * i + 1] * inv_s) << 4);
    }
}

__global__ void quant_bf16_to_fp8_kernel(
    const __nv_bfloat16* __restrict__ src, uint8_t* __restrict__ dst_w, float* __restrict__ dst_row_scales,
    int N, int K) {
    int r = blockIdx.x;
    if (r >= N) return;
    const __nv_bfloat16* row_in = src + (size_t)r * K;
    float lmax = 0.0f;
    for (int c = threadIdx.x; c < K; c += blockDim.x) lmax = fmaxf(lmax, fabsf(__bfloat162float(row_in[c])));
    lmax = warp_max(lmax);
    __shared__ float smem[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = lmax;
    __syncthreads();
    __shared__ float inv_scale;
    if (warp == 0) {
        float v = warp_max((lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f);
        if (lane == 0) {
            float sc = fmaxf(v / 448.0f, 1e-12f);
            dst_row_scales[r] = sc;
            inv_scale = 1.0f / sc;
        }
    }
    __syncthreads();
    float isc = inv_scale;
    uint8_t* row_out = dst_w + (size_t)r * K;
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        __nv_fp8_e4m3 q(__bfloat162float(row_in[c]) * isc);
        row_out[c] = q.__x;
    }
}

__global__ void mxfp8_expert_amax_kernel(
    const uint8_t* __restrict__ src_w, const uint8_t* __restrict__ src_s,
    float* __restrict__ exp_amax, int rows, int cols) {
    int e = blockIdx.x;
    size_t expert_elems = (size_t)rows * cols;
    size_t n_blocks32 = expert_elems >> 5;
    const uint4* ew4 = reinterpret_cast<const uint4*>(src_w + (size_t)e * expert_elems);
    const uint8_t* es = src_s + (size_t)e * n_blocks32;
    float lmax = 0.0f;
    for (size_t blk = threadIdx.x; blk < n_blocks32; blk += blockDim.x) {
        float sc = cvt_ue8m0_to_float(__ldg(es + blk));
        uint4 q0 = __ldg(ew4 + blk * 2);
        uint4 q1 = __ldg(ew4 + blk * 2 + 1);
        const uint32_t words[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        float bmax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float2 a = __half22float2(cvt_e4m3x2_to_half2(static_cast<uint16_t>(words[i] & 0xFFFFu)));
            float2 b = __half22float2(cvt_e4m3x2_to_half2(static_cast<uint16_t>(words[i] >> 16)));
            bmax = fmaxf(bmax, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y))));
        }
        lmax = fmaxf(lmax, bmax * sc);
    }
    lmax = warp_max(lmax);
    __shared__ float smem[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = lmax;
    __syncthreads();
    if (warp == 0) {
        float v = warp_max((lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f);
        if (lane == 0) exp_amax[e] = fmaxf(v / (448.0f * 6.0f), 1e-12f);
    }
}

__global__ void mxfp8_to_nvfp4_expert_kernel(
    const uint8_t* __restrict__ src_w, const uint8_t* __restrict__ src_s,
    const float* __restrict__ exp_global_scale, uint8_t* __restrict__ dst_w, uint8_t* __restrict__ dst_s,
    int n_experts, int rows, int cols) {
    size_t blocks_per_expert = (size_t)rows * (cols / 16);
    size_t total_blocks = (size_t)n_experts * blocks_per_expert;
    size_t bidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bidx >= total_blocks) return;
    int e = (int)(bidx / blocks_per_expert);
    float gscale = exp_global_scale[e];
    size_t elem_off = bidx * 16;
    float mx_scale = cvt_ue8m0_to_float(src_s[elem_off >> 5]);
    float vals[16];
    float bmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] = cvt_e4m3_to_float(src_w[elem_off + i]) * mx_scale;
        bmax = fmaxf(bmax, fabsf(vals[i]));
    }
    __nv_fp8_e4m3 s_fp8(fminf(bmax / (6.0f * gscale), 448.0f));
    dst_s[bidx] = s_fp8.__x;
    float s_eff = static_cast<float>(s_fp8) * gscale;
    float inv_s = (s_eff > 1e-30f) ? (1.0f / s_eff) : 0.0f;
    uint8_t* out_w = dst_w + bidx * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_w[i] = quantize_e2m1(vals[2 * i] * inv_s) | (quantize_e2m1(vals[2 * i + 1] * inv_s) << 4);
    }
}

__global__ void compute_dspark_fold_bias_kernel(
    const __nv_bfloat16* __restrict__ unembed, const __nv_bfloat16* __restrict__ hidden_out_b,
    float* __restrict__ fold_b, int V, int K) {
    int v = blockIdx.x * blockDim.y + threadIdx.y;
    if (v >= V) return;
    int lane = threadIdx.x;
    const __nv_bfloat16* row = unembed + (size_t)v * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) acc = fmaf(__bfloat162float(row[k]), __bfloat162float(hidden_out_b[k]), acc);
    acc = warp_sum(acc);
    if (lane == 0) fold_b[v] = acc;
}

__global__ void compute_dspark_fold_w_kernel(
    const __nv_bfloat16* __restrict__ unembed, const __nv_bfloat16* __restrict__ hidden_out_w,
    __nv_bfloat16* __restrict__ fold_out, int V, int K, int R) {
    int v = blockIdx.x;
    if (v >= V) return;
    int r = threadIdx.x;
    __shared__ float u_row[HIDDEN];
    const __nv_bfloat16* u_ptr = unembed + (size_t)v * K;
    for (int k = r; k < K; k += 256) u_row[k] = __bfloat162float(u_ptr[k]);
    __syncthreads();
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc = fmaf(u_row[k], __bfloat162float(hidden_out_w[(size_t)k * R + r]), acc);
    fold_out[(size_t)v * R + r] = __float2bfloat16(acc);
}

__global__ void split_dspark_rnn_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ w_sp, __nv_bfloat16* __restrict__ w_h, __nv_bfloat16* __restrict__ w_f) {
    int o = blockIdx.x;
    constexpr int K_IN = 2 * MTP_MARKOV_RANK + 2 * HIDDEN;
    const __nv_bfloat16* row = src + (size_t)o * K_IN;
    for (int c = threadIdx.x; c < K_IN; c += 256) {
        __nv_bfloat16 v = row[c];
        if (c < 512) w_sp[(size_t)o * 512 + c] = v;
        else if (c < 512 + HIDDEN) w_h[(size_t)o * HIDDEN + (c - 512)] = v;
        else w_f[(size_t)o * HIDDEN + (c - 512 - HIDDEN)] = v;
    }
}

__global__ void inc_pos_and_copy_tok_kernel(
    int32_t* d_pos, int32_t* d_tok_in, const int32_t* d_tok_out, int32_t* result) {
    const int32_t token = *d_tok_out;
    *d_pos += 1;
    *d_tok_in = token;
    *result = token;
}

__global__ void inc_pos_by_kernel(int32_t* d_pos, int delta) {
    if (threadIdx.x == 0) *d_pos += delta;
}

__global__ void assemble_verify_tokens_kernel(
    const int32_t* d_anchor, const int32_t* d_drafts, int32_t* d_tokens, int num_drafts) {
    int i = threadIdx.x;
    if (i == 0) d_tokens[0] = *d_anchor;
    else if (i <= num_drafts) d_tokens[i] = d_drafts[i - 1];
}

__global__ void adaptive_select_width_kernel(
    const int32_t* __restrict__ d_anchor,
    const int32_t* __restrict__ d_drafts,
    const float* __restrict__ d_draft_conf,
    float threshold,
    int32_t* __restrict__ d_tokens,
    int32_t* __restrict__ d_chosen_width,
    cudaGraphConditionalHandle cond_handle) {
    if (threadIdx.x != 0) return;
    // Strict per-token 5-branch {1, 2, 3, 4, 8} selector tuned to GB10 MoE bandwidth cliff.
    const float th1 = 0.85f * threshold;
    unsigned int case_idx = 0u;
    int width = 1;
    if (d_draft_conf[0] >= th1) {
        case_idx = 1u; width = 2;
        if (d_draft_conf[0] >= threshold && d_draft_conf[1] >= threshold) {
            case_idx = 2u; width = 3;
            if (d_draft_conf[2] >= threshold) {
                case_idx = 3u; width = 4;
                if (d_draft_conf[3] >= threshold && d_draft_conf[4] >= threshold) {
                    case_idx = 4u; width = 8;
                }
            }
        }
    }
    d_tokens[0] = *d_anchor;
    for (int i = 0; i < width - 1; ++i) d_tokens[1 + i] = d_drafts[i];
    *d_chosen_width = width;
    cudaGraphSetConditional(cond_handle, case_idx);
}

} // namespace

struct AdaptiveGraph {
    cudaGraph_t root_graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraphConditionalHandle handle{};
    cudaGraph_t bodies[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
};

void launch_bf16_amax_accum(
    const void* src_bf16_dev, size_t elems, float* d_amax_scratch, cudaStream_t stream) {
    int grid = (int)std::min<size_t>((elems + 255) / 256, 2048);
    bf16_amax_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src_bf16_dev), elems, d_amax_scratch);
}

void launch_pack_bf16_to_nvfp4_with_scale(
    const void* src_bf16_dev, void* dst_data, uint8_t* dst_scales,
    float global_scale, int32_t rows, int32_t cols, cudaStream_t stream) {
    size_t total_blocks = (size_t)rows * (cols / 16);
    quant_bf16_to_nvfp4_kernel<<<(int)((total_blocks + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src_bf16_dev),
        static_cast<uint8_t*>(dst_data), dst_scales, rows, cols, global_scale);
}

float pack_bf16_to_nvfp4(
    const void* src_bf16_dev, void* dst_data, uint8_t* dst_scales, float* d_amax_scratch,
    int32_t rows, int32_t cols) {
    CUDA_CHECK(cudaMemset(d_amax_scratch, 0, sizeof(float)));
    size_t n = (size_t)rows * cols;
    int grid = (int)std::min<size_t>((n + 255) / 256, 2048);
    bf16_amax_kernel<<<grid, 256>>>(static_cast<const __nv_bfloat16*>(src_bf16_dev), n, d_amax_scratch);
    float h_amax = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_amax, d_amax_scratch, sizeof(float), cudaMemcpyDeviceToHost));
    float gscale = std::max(h_amax / (448.0f * 6.0f), 1e-12f);
    size_t total_blocks = (size_t)rows * (cols / 16);
    quant_bf16_to_nvfp4_kernel<<<(int)((total_blocks + 255) / 256), 256>>>(
        static_cast<const __nv_bfloat16*>(src_bf16_dev),
        static_cast<uint8_t*>(dst_data), dst_scales, rows, cols, gscale);
    return gscale;
}

void pack_bf16_to_fp8(
    const void* src_bf16_dev, void* dst_data, float* dst_row_scales, int32_t rows, int32_t cols) {
    quant_bf16_to_fp8_kernel<<<rows, 256>>>(
        static_cast<const __nv_bfloat16*>(src_bf16_dev),
        static_cast<uint8_t*>(dst_data), dst_row_scales, rows, cols);
}

void convert_mxfp8_to_nvfp4_experts(
    const uint8_t* src_w, const uint8_t* src_s,
    uint8_t* dst_w, uint8_t* dst_s, float* dst_expert_scale,
    int32_t n_experts, int32_t rows, int32_t cols) {
    mxfp8_expert_amax_kernel<<<n_experts, 256>>>(src_w, src_s, dst_expert_scale, rows, cols);
    size_t total_blocks = (size_t)n_experts * rows * (cols / 16);
    mxfp8_to_nvfp4_expert_kernel<<<(int)((total_blocks + 255) / 256), 256>>>(
        src_w, src_s, dst_expert_scale, dst_w, dst_s, n_experts, rows, cols);
}

void split_dspark_rnn(const void* src_bf16, void* w_sp, void* w_h, void* w_f) {
    split_dspark_rnn_kernel<<<768, 256>>>(
        static_cast<const __nv_bfloat16*>(src_bf16),
        static_cast<__nv_bfloat16*>(w_sp),
        static_cast<__nv_bfloat16*>(w_h),
        static_cast<__nv_bfloat16*>(w_f));
}

void compute_dspark_fold(
    const void* unembed_bf16, const void* hidden_out_w, const void* hidden_out_b,
    float* fold_b, void* fold_bf16_out, int32_t V, int32_t K, int32_t R) {
    dim3 fb_blk(32, 8);
    compute_dspark_fold_bias_kernel<<<(V + 7) / 8, fb_blk>>>(
        static_cast<const __nv_bfloat16*>(unembed_bf16),
        static_cast<const __nv_bfloat16*>(hidden_out_b), fold_b, V, K);

    static cublasHandle_t fold_cublas = nullptr;
    if (!fold_cublas) {
        if (cublasCreate(&fold_cublas) == CUBLAS_STATUS_SUCCESS) {
            cublasSetMathMode(fold_cublas, CUBLAS_DEFAULT_MATH);
        }
    }
    if (fold_cublas) {
        float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(
            fold_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            R, V, K,
            &alpha,
            hidden_out_w, CUDA_R_16BF, R,
            unembed_bf16, CUDA_R_16BF, K,
            &beta,
            fold_bf16_out, CUDA_R_16BF, R,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st == CUBLAS_STATUS_SUCCESS) return;
    }
    compute_dspark_fold_w_kernel<<<V, 256>>>(
        static_cast<const __nv_bfloat16*>(unembed_bf16),
        static_cast<const __nv_bfloat16*>(hidden_out_w),
        static_cast<__nv_bfloat16*>(fold_bf16_out), V, K, R);
}

void launch_inc_pos_and_copy_tok(
    int32_t* d_pos, int32_t* d_tok_in, const int32_t* d_tok_out, int32_t* result, cudaStream_t stream) {
    inc_pos_and_copy_tok_kernel<<<1, 1, 0, stream>>>(d_pos, d_tok_in, d_tok_out, result);
}

void launch_inc_pos_by(int32_t* d_pos, int32_t delta, cudaStream_t stream) {
    inc_pos_by_kernel<<<1, 1, 0, stream>>>(d_pos, delta);
}

void launch_assemble_verify_tokens(
    const int32_t* d_anchor, const int32_t* d_drafts, int32_t* d_tokens, int32_t num_drafts, cudaStream_t stream) {
    assemble_verify_tokens_kernel<<<1, num_drafts + 1, 0, stream>>>(d_anchor, d_drafts, d_tokens, num_drafts);
}

cudaError_t adaptive_graph_create(
    const int32_t* d_anchor, const int32_t* d_drafts, const float* d_draft_conf, float threshold,
    int32_t* d_tokens, int32_t* d_chosen_width, cudaStream_t stream, AdaptiveGraph** out_graph) {
    AdaptiveGraph* ag = new AdaptiveGraph();
    cudaError_t err = cudaGraphCreate(&ag->root_graph, 0);
    if (err != cudaSuccess) { delete ag; return err; }
    err = cudaGraphConditionalHandleCreate(&ag->handle, ag->root_graph, 0, 0);
    if (err != cudaSuccess) { cudaGraphDestroy(ag->root_graph); delete ag; return err; }

    err = cudaStreamBeginCaptureToGraph(stream, ag->root_graph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) { cudaGraphDestroy(ag->root_graph); delete ag; return err; }

    adaptive_select_width_kernel<<<1, 1, 0, stream>>>(
        d_anchor, d_drafts, d_draft_conf, threshold, d_tokens, d_chosen_width, ag->handle);

    cudaStreamCaptureStatus status;
    cudaGraph_t g = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    const cudaGraphEdgeData* edges = nullptr;
    size_t ndeps = 0;
    err = cudaStreamGetCaptureInfo(stream, &status, nullptr, &g, &deps, &edges, &ndeps);
    if (err != cudaSuccess) {
        cudaGraph_t disc = nullptr; cudaStreamEndCapture(stream, &disc);
        cudaGraphDestroy(ag->root_graph); delete ag; return err;
    }

    cudaGraphNodeParams p{};
    p.type = cudaGraphNodeTypeConditional;
    p.conditional.handle = ag->handle;
    p.conditional.type = cudaGraphCondTypeSwitch;
    p.conditional.size = 5;
    cudaGraphNode_t sw_node = nullptr;
    err = cudaGraphAddNode(&sw_node, g, deps, edges, ndeps, &p);
    if (err != cudaSuccess) {
        cudaGraph_t disc = nullptr; cudaStreamEndCapture(stream, &disc);
        cudaGraphDestroy(ag->root_graph); delete ag; return err;
    }
    for (int bi = 0; bi < 5; ++bi) ag->bodies[bi] = p.conditional.phGraph_out[bi];

    err = cudaStreamUpdateCaptureDependencies(stream, &sw_node, nullptr, 1, cudaStreamSetCaptureDependencies);
    if (err != cudaSuccess) {
        cudaGraph_t disc = nullptr; cudaStreamEndCapture(stream, &disc);
        cudaGraphDestroy(ag->root_graph); delete ag; return err;
    }

    cudaGraph_t out_g = nullptr;
    err = cudaStreamEndCapture(stream, &out_g);
    if (err != cudaSuccess) { cudaGraphDestroy(ag->root_graph); delete ag; return err; }
    *out_graph = ag;
    return cudaSuccess;
}

cudaError_t adaptive_graph_branch_begin(AdaptiveGraph* graph, int32_t branch_idx, cudaStream_t stream) {
    return cudaStreamBeginCaptureToGraph(
        stream, graph->bodies[branch_idx], nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal);
}

cudaError_t adaptive_graph_branch_end(cudaStream_t stream) {
    cudaGraph_t out_g = nullptr;
    return cudaStreamEndCapture(stream, &out_g);
}

cudaError_t adaptive_graph_instantiate(AdaptiveGraph* graph) {
    cudaError_t err = cudaGraphInstantiate(&graph->exec, graph->root_graph, 0);
    if (graph->root_graph) {
        cudaGraphDestroy(graph->root_graph);
        graph->root_graph = nullptr;
    }
    return err;
}

cudaError_t adaptive_graph_launch(AdaptiveGraph* graph, cudaStream_t stream) {
    return cudaGraphLaunch(graph->exec, stream);
}

void adaptive_graph_destroy(AdaptiveGraph* graph) {
    if (!graph) return;
    if (graph->exec) cudaGraphExecDestroy(graph->exec);
    if (graph->root_graph) cudaGraphDestroy(graph->root_graph);
    delete graph;
}

namespace {
__global__ void assemble_batched_verify_tokens_kernel(
    const BatchSeqMeta* __restrict__ seq_meta,
    int32_t num_seqs,
    const int32_t* __restrict__ slot_anchors,
    const int32_t* __restrict__ slot_drafts,
    int32_t* __restrict__ out_tokens,
    int32_t* __restrict__ out_draft_tokens) {
    if (threadIdx.x != 0) return;
    for (int32_t si = 0; si < num_seqs; ++si) {
        const BatchSeqMeta sm = seq_meta[si];
        out_tokens[sm.batch_off] = slot_anchors[sm.slot_id];
        for (int32_t j = 0; j < sm.width - 1; ++j) {
            const int32_t d = slot_drafts[sm.slot_id * MTP_MAX_DRAFTS + j];
            out_tokens[sm.batch_off + 1 + j] = d;
            out_draft_tokens[sm.batch_off + j] = d;
        }
        out_draft_tokens[sm.batch_off + sm.width - 1] = -1;
    }
}
} // namespace

void launch_assemble_batched_verify_tokens(
    const BatchSeqMeta* d_seq_meta,
    int32_t num_seqs,
    const int32_t* d_slot_anchors,
    const int32_t* d_slot_drafts,
    int32_t* d_out_tokens,
    int32_t* d_out_draft_tokens,
    cudaStream_t stream) {
    assemble_batched_verify_tokens_kernel<<<1, 1, 0, stream>>>(
        d_seq_meta, num_seqs, d_slot_anchors, d_slot_drafts, d_out_tokens, d_out_draft_tokens);
}

namespace {
__global__ void arch_probe_kernel() {}
} // namespace

bool check_gpu_supported() {
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        fprintf(stderr, "[gpu] No CUDA device available (driver missing or GPU not passed to the container?)\n");
        return false;
    }
    printf("[gpu] %s | sm_%d%d | %d SMs | %.1f GiB %s\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
           prop.integrated ? "unified memory" : "VRAM");
    if (prop.major < 8) {
        fprintf(stderr, "[gpu] %s (sm_%d%d) is not supported: squidDGX needs compute capability 8.0+ "
                        "(BF16 tensor cores, cp.async). Default build targets: A10G (sm_86), L4/L40S (sm_89), "
                        "H100 (Hopper), RTX PRO Blackwell, DGX Spark.\n",
                prop.name, prop.major, prop.minor);
        return false;
    }
    cudaFuncAttributes attr{};
    const cudaError_t err = cudaFuncGetAttributes(&attr, arch_probe_kernel);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "[gpu] This build has no kernels for sm_%d%d (%s). Rebuild with "
                        "CUDA_ARCHS including %d%d%s (docker build --build-arg CUDA_ARCHS=...).\n",
                prop.major, prop.minor, cudaGetErrorString(err), prop.major, prop.minor,
                prop.major >= 10 ? "a" : "");
        return false;
    }
    return true;
}

} // namespace squid
