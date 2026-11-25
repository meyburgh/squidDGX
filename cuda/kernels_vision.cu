#include "kernels.hpp"
#include "kernels_common.cuh"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>

namespace squid {

namespace {

static cublasHandle_t get_vis_cublas() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasStatus_t st = cublasCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[vision] cublasCreate failed: %d\n", static_cast<int>(st));
            std::exit(1);
        }
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    return handle;
}

// Row-major BF16 GEMM: Y[M, N] = X[M, K] @ W[N, K]^T (FP32 accumulation on Tensor Cores)
static void vis_gemm_bf16(
    cublasHandle_t h,
    const __nv_bfloat16* x,
    const __nv_bfloat16* w,
    __nv_bfloat16* y,
    int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        h,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        w, CUDA_R_16BF, K,
        x, CUDA_R_16BF, K,
        &beta,
        y, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[vision] cublasGemmEx failed: %d (M=%d N=%d K=%d)\n",
                static_cast<int>(st), M, N, K);
        std::exit(1);
    }
}

// Populate CLS token at X[0] and add bilinearly interpolated 2D position embeddings to X[1..N_patch].
// Grid = (N_patch + 1), Block = 256 (each thread handles 4 channels of VIS_HIDDEN=1024).
__global__ void vis_embed_add_posemb_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ class_embed,
    const __nv_bfloat16* __restrict__ cls_pos_embed,
    const __nv_bfloat16* __restrict__ pos_embed,
    int grid_rows,
    int grid_cols) {
    const int s = blockIdx.x;
    const int tid = threadIdx.x;

    if (s == 0) {
        for (int c = tid; c < VIS_HIDDEN; c += blockDim.x) {
            float v = __bfloat162float(class_embed[c]) + __bfloat162float(cls_pos_embed[c]);
            x[c] = __float2bfloat16(v);
        }
        return;
    }

    const int i = s - 1;
    const int tok = i >> 2;
    const int sub = i & 3;
    const int br = tok / grid_cols;
    const int bc = tok % grid_cols;
    const int row = br * VIS_MERGE_SIZE + (sub >> 1);
    const int col = bc * VIS_MERGE_SIZE + (sub & 1);
    const int hp = grid_rows * VIS_MERGE_SIZE;
    const int wp = grid_cols * VIS_MERGE_SIZE;

    constexpr int SIDE = VIS_POSEMB_GRID;
    const float src_h = (static_cast<float>(row) + 0.5f) * static_cast<float>(SIDE) / static_cast<float>(hp) - 0.5f;
    const float src_w = (static_cast<float>(col) + 0.5f) * static_cast<float>(SIDE) / static_cast<float>(wp) - 0.5f;
    const float fh = floorf(src_h);
    const float fw = floorf(src_w);
    const int ih = static_cast<int>(fh);
    const int iw = static_cast<int>(fw);
    const int h0 = max(0, min(SIDE - 1, ih));
    const int h1 = max(0, min(SIDE - 1, ih + 1));
    const int w0 = max(0, min(SIDE - 1, iw));
    const int w1 = max(0, min(SIDE - 1, iw + 1));
    const float dh = src_h - fh;
    const float dw = src_w - fw;
    const float wh0 = 1.0f - dh;
    const float wh1 = dh;
    const float ww0 = 1.0f - dw;
    const float ww1 = dw;

    const float wt00 = wh0 * ww0;
    const float wt01 = wh0 * ww1;
    const float wt10 = wh1 * ww0;
    const float wt11 = wh1 * ww1;

    const __nv_bfloat16* p00 = pos_embed + (size_t)(h0 * SIDE + w0) * VIS_HIDDEN;
    const __nv_bfloat16* p01 = pos_embed + (size_t)(h0 * SIDE + w1) * VIS_HIDDEN;
    const __nv_bfloat16* p10 = pos_embed + (size_t)(h1 * SIDE + w0) * VIS_HIDDEN;
    const __nv_bfloat16* p11 = pos_embed + (size_t)(h1 * SIDE + w1) * VIS_HIDDEN;
    __nv_bfloat16* dst = x + (size_t)s * VIS_HIDDEN;

    for (int c = tid; c < VIS_HIDDEN; c += blockDim.x) {
        float pe = wt00 * __bfloat162float(p00[c])
                 + wt01 * __bfloat162float(p01[c])
                 + wt10 * __bfloat162float(p10[c])
                 + wt11 * __bfloat162float(p11[c]);
        dst[c] = __float2bfloat16(__bfloat162float(dst[c]) + pe);
    }
}

// Exact FP32 LayerNorm (eps=1e-5) for VIS_HIDDEN=1024: BF16 -> BF16.
// Grid = S, Block = 256.
__global__ void vis_layernorm_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out) {
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const __nv_bfloat16* xp = in + (size_t)s * VIS_HIDDEN;
    __nv_bfloat16* yp = out + (size_t)s * VIS_HIDDEN;

    float v[4];
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int c = tid + k * 256;
        v[k] = __bfloat162float(xp[c]);
        sum += v[k];
    }
    sum = warp_sum(sum);
    __shared__ float sm[8];
    if (lane == 0) sm[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < 8) ? sm[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) sm[0] = t * (1.0f / static_cast<float>(VIS_HIDDEN));
    }
    __syncthreads();
    const float mean = sm[0];

    float var_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        float d = v[k] - mean;
        v[k] = d;
        var_sum += d * d;
    }
    var_sum = warp_sum(var_sum);
    if (lane == 0) sm[warp] = var_sum;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < 8) ? sm[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) sm[0] = rsqrtf(t * (1.0f / static_cast<float>(VIS_HIDDEN)) + VIS_LN_EPS);
    }
    __syncthreads();
    const float inv_std = sm[0];

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int c = tid + k * 256;
        float res = v[k] * inv_std * __bfloat162float(weight[c]) + __bfloat162float(bias[c]);
        yp[c] = __float2bfloat16(res);
    }
}

// Add QKV bias, apply exact Axial 2D RoPE to Q & K (width on channels 0..31, height on 32..63),
// and transpose Q, K, V from [S, 3*1024] -> [H=16, S, D=64] BF16!
// Grid = S, Block = 512 (16 heads * 32 pairs = 512 threads).
__global__ void vis_add_bias_rope_qkv_kernel(
    const __nv_bfloat16* __restrict__ qkv_in,  // [S, 3072]
    const __nv_bfloat16* __restrict__ qkv_b,   // [3072]
    __nv_bfloat16* __restrict__ q_out,         // [16, S, 64]
    __nv_bfloat16* __restrict__ k_out,         // [16, S, 64]
    __nv_bfloat16* __restrict__ v_out,         // [16, S, 64]
    int S,
    int grid_cols) {
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    const int h = tid >> 5;   // 0..15
    const int p = tid & 31;   // 0..31 (pair of channels 2*p, 2*p+1)

    float pos_h = 0.0f;
    float pos_w = 0.0f;
    if (s > 0) {
        const int i = s - 1;
        const int tok = i >> 2;
        const int sub = i & 3;
        const int br = tok / grid_cols;
        const int bc = tok % grid_cols;
        const int row = br * VIS_MERGE_SIZE + (sub >> 1);
        const int col = bc * VIS_MERGE_SIZE + (sub & 1);
        pos_h = static_cast<float>(row + 1);
        pos_w = static_cast<float>(col + 1);
    }

    // Axial 2D RoPE: channels 0..31 (p=0..15) use pos_w; channels 32..63 (p=16..31) use pos_h.
    const int freq_idx = p & 15;
    const float inv_freq = __powf(VIS_ROPE_THETA, -static_cast<float>(2 * freq_idx) / static_cast<float>(VIS_HEAD_DIM / 2));
    const float angle = ((p < 16) ? pos_w : pos_h) * inv_freq;
    float sn, cs;
    __sincosf(angle, &sn, &cs);

    const int d0 = 2 * p;
    const int d1 = 2 * p + 1;
    const int c0 = h * VIS_HEAD_DIM + d0;
    const int c1 = h * VIS_HEAD_DIM + d1;
    const __nv_bfloat16* row_in = qkv_in + (size_t)s * (3 * VIS_HIDDEN);

    float q0 = __bfloat162float(row_in[c0]) + __bfloat162float(qkv_b[c0]);
    float q1 = __bfloat162float(row_in[c1]) + __bfloat162float(qkv_b[c1]);
    float k0 = __bfloat162float(row_in[VIS_HIDDEN + c0]) + __bfloat162float(qkv_b[VIS_HIDDEN + c0]);
    float k1 = __bfloat162float(row_in[VIS_HIDDEN + c1]) + __bfloat162float(qkv_b[VIS_HIDDEN + c1]);
    float v0 = __bfloat162float(row_in[2 * VIS_HIDDEN + c0]) + __bfloat162float(qkv_b[2 * VIS_HIDDEN + c0]);
    float v1 = __bfloat162float(row_in[2 * VIS_HIDDEN + c1]) + __bfloat162float(qkv_b[2 * VIS_HIDDEN + c1]);

    const size_t dst_off = ((size_t)h * S + s) * VIS_HEAD_DIM;
    q_out[dst_off + d0] = __float2bfloat16(q0 * cs - q1 * sn);
    q_out[dst_off + d1] = __float2bfloat16(q1 * cs + q0 * sn);
    k_out[dst_off + d0] = __float2bfloat16(k0 * cs - k1 * sn);
    k_out[dst_off + d1] = __float2bfloat16(k1 * cs + k0 * sn);
    v_out[dst_off + d0] = __float2bfloat16(v0);
    v_out[dst_off + d1] = __float2bfloat16(v1);
}

// Bidirectional tiled FlashAttention for VIS_HEAD_DIM=64 (O(S) memory, FP32 online softmax).
// Grid = dim3((S + 3) / 4, VIS_HEADS=16), Block = 128 (4 warps, 1 warp per query row).
__global__ void __launch_bounds__(128)
vis_flash_attn_kernel(
    const __nv_bfloat16* __restrict__ q,   // [16, S, 64]
    const __nv_bfloat16* __restrict__ k,   // [16, S, 64]
    const __nv_bfloat16* __restrict__ v,   // [16, S, 64]
    __nv_bfloat16* __restrict__ out,       // [S, 1024]
    int S) {
    constexpr int BR = 4;
    constexpr int BC = 32;
    const int head = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q_idx = blockIdx.x * BR + warp;

    const __nv_bfloat16* q_head = q + (size_t)head * S * VIS_HEAD_DIM;
    const __nv_bfloat16* k_head = k + (size_t)head * S * VIS_HEAD_DIM;
    const __nv_bfloat16* v_head = v + (size_t)head * S * VIS_HEAD_DIM;

    constexpr float scale = 0.125f; // 1 / sqrt(64)
    float q0 = 0.0f, q1 = 0.0f;
    if (q_idx < S) {
        q0 = __bfloat162float(q_head[(size_t)q_idx * VIS_HEAD_DIM + lane]) * scale;
        q1 = __bfloat162float(q_head[(size_t)q_idx * VIS_HEAD_DIM + lane + 32]) * scale;
    }

    __shared__ __nv_bfloat16 sK[BC][VIS_HEAD_DIM];
    __shared__ __nv_bfloat16 sV[BC][VIS_HEAD_DIM];

    float m_i = -1e30f;
    float l_i = 0.0f;
    float o0 = 0.0f, o1 = 0.0f;

    for (int j0 = 0; j0 < S; j0 += BC) {
        const int tile_len = min(BC, S - j0);
        // Collaborative load of [tile_len, 64] K and V into SMEM (128 threads * 16 elems = 2048 elems).
        for (int idx = threadIdx.x; idx < tile_len * VIS_HEAD_DIM; idx += 128) {
            const int r = idx >> 6;
            const int d = idx & 63;
            sK[r][d] = k_head[(size_t)(j0 + r) * VIS_HEAD_DIM + d];
            sV[r][d] = v_head[(size_t)(j0 + r) * VIS_HEAD_DIM + d];
        }
        __syncthreads();

        if (q_idx < S) {
            for (int r = 0; r < tile_len; ++r) {
                float dot = q0 * __bfloat162float(sK[r][lane]) + q1 * __bfloat162float(sK[r][lane + 32]);
                dot = warp_sum(dot);
                const float m_new = fmaxf(m_i, dot);
                const float alpha = __expf(m_i - m_new);
                const float p = __expf(dot - m_new);
                l_i = l_i * alpha + p;
                o0 = o0 * alpha + p * __bfloat162float(sV[r][lane]);
                o1 = o1 * alpha + p * __bfloat162float(sV[r][lane + 32]);
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (q_idx < S) {
        const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        __nv_bfloat16* dst = out + (size_t)q_idx * VIS_HIDDEN + head * VIS_HEAD_DIM;
        dst[lane]      = __float2bfloat16(o0 * inv_l);
        dst[lane + 32] = __float2bfloat16(o1 * inv_l);
    }
}

// Fused bias + LayerScale + residual: x[s, c] = bf16(float(x[s, c]) + float(gamma[c]) * (float(y[s, c]) + float(bias[c]))).
__global__ void vis_bias_layerscale_res_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ gamma,
    int total_elems) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    const int c = idx & (VIS_HIDDEN - 1);
    float val = __bfloat162float(x[idx])
              + __bfloat162float(gamma[c]) * (__bfloat162float(y[idx]) + __bfloat162float(bias[c]));
    x[idx] = __float2bfloat16(val);
}

// Fused optional bias + exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
__global__ void vis_bias_gelu_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias, // optional (nullptr for proj_in)
    int width,
    int total_elems) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    float v = __bfloat162float(x[idx]);
    if (bias) v += __bfloat162float(bias[idx % width]);
    v = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
    x[idx] = __float2bfloat16(v);
}

__global__ void vis_bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ in,
    float* __restrict__ out,
    int total_elems) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    out[idx] = __bfloat162float(in[idx]);
}

// Scatter encoded vision embeddings into LLM hidden states after embed_norm.
// Grid = n_slots, Block = 256 (each thread copies 9 floats of HIDDEN=2304).
__global__ void scatter_vision_embeds_kernel(
    float* __restrict__ hidden,            // [B, HIDDEN]
    const float* __restrict__ vis_embeds,  // [total_img_tokens, HIDDEN]
    const int32_t* __restrict__ dst_rows,  // [n_slots] batch index in 0..B-1
    const int32_t* __restrict__ src_rows,  // [n_slots] image embed index
    int n_slots) {
    const int slot = blockIdx.x;
    if (slot >= n_slots) return;
    const int dst_b = dst_rows[slot];
    const int src_i = src_rows[slot];
    const float4* src = reinterpret_cast<const float4*>(vis_embeds + (size_t)src_i * HIDDEN);
    float4* dst = reinterpret_cast<float4*>(hidden + (size_t)dst_b * HIDDEN);
    constexpr int VEC_LEN = HIDDEN / 4; // 576
    for (int c = threadIdx.x; c < VEC_LEN; c += blockDim.x) {
        dst[c] = src[c];
    }
}

} // namespace

void launch_vision_encode(
    const VisionWeights& vw,
    const __nv_bfloat16* d_patches,
    int grid_rows,
    int grid_cols,
    float* d_out_embeds,
    cudaStream_t stream) {
    const int n_tok = grid_rows * grid_cols;
    const int n_patch = n_tok * VIS_MERGE_SIZE * VIS_MERGE_SIZE;
    const int S = n_patch + 1;

    cublasHandle_t h = get_vis_cublas();
    cublasSetStream(h, stream);

    // Allocate compact temporary workspace (freed on exit -> 0 permanent VRAM overhead).
    const size_t x_bytes    = (size_t)S * VIS_HIDDEN * sizeof(__nv_bfloat16);
    const size_t qkv_bytes  = (size_t)S * (3 * VIS_HIDDEN) * sizeof(__nv_bfloat16);
    const size_t mlp_bytes  = (size_t)S * VIS_INTERMEDIATE * sizeof(__nv_bfloat16);
    const size_t proj_bytes = (size_t)n_tok * VIS_PROJ_HIDDEN * sizeof(__nv_bfloat16);
    const size_t out_b_bytes = (size_t)n_tok * HIDDEN * sizeof(__nv_bfloat16);
    const size_t big_tmp    = max(max(qkv_bytes, mlp_bytes), proj_bytes);

    auto align256 = [](size_t v) -> size_t { return (v + 255) & ~size_t(255); };
    const size_t off_x    = 0;
    const size_t off_h    = off_x + align256(x_bytes);
    const size_t off_tmp  = off_h + align256(x_bytes);
    const size_t off_q    = off_tmp + align256(big_tmp);
    const size_t off_k    = off_q + align256(x_bytes);
    const size_t off_v    = off_k + align256(x_bytes);
    const size_t off_attn = off_v + align256(x_bytes);
    const size_t total_ws = off_attn + align256(max(x_bytes, out_b_bytes));

    uint8_t* ws = nullptr;
    CUDA_CHECK(cudaMalloc(&ws, total_ws));

    auto* d_x    = reinterpret_cast<__nv_bfloat16*>(ws + off_x);
    auto* d_h    = reinterpret_cast<__nv_bfloat16*>(ws + off_h);
    auto* d_tmp  = reinterpret_cast<__nv_bfloat16*>(ws + off_tmp);
    auto* d_q    = reinterpret_cast<__nv_bfloat16*>(ws + off_q);
    auto* d_k    = reinterpret_cast<__nv_bfloat16*>(ws + off_k);
    auto* d_v    = reinterpret_cast<__nv_bfloat16*>(ws + off_v);
    auto* d_attn = reinterpret_cast<__nv_bfloat16*>(ws + off_attn);

    // 1. Patch embedding: X[1..N_patch] = patches[N_patch, 1200] @ patch_embedding[1024, 1200]^T
    vis_gemm_bf16(h, d_patches, vw.patch_embedding, d_x + VIS_HIDDEN, n_patch, VIS_HIDDEN, VIS_PATCH_DIM);

    // 2. CLS token + bilinear interpolated 2D position embeddings
    vis_embed_add_posemb_kernel<<<S, 256, 0, stream>>>(
        d_x, vw.class_embedding, vw.cls_position_embedding, vw.position_embedding, grid_rows, grid_cols);

    // 3. Pre-LayerNorm (in-place via d_h -> swap or write back to d_x)
    vis_layernorm_kernel<<<S, 256, 0, stream>>>(d_x, vw.ln_pre_w, vw.ln_pre_b, d_h);
    CUDA_CHECK(cudaMemcpyAsync(d_x, d_h, x_bytes, cudaMemcpyDeviceToDevice, stream));

    const int x_elems = S * VIS_HIDDEN;
    const int mlp_elems = S * VIS_INTERMEDIATE;
    const int x_blks = (x_elems + 255) / 256;
    const int mlp_blks = (mlp_elems + 255) / 256;

    // 4. 23 ViT Encoder Layers
    for (int l = 0; l < VIS_LAYERS; ++l) {
        const VisionLayerWeights& lw = vw.layers[l];

        // Attention block
        vis_layernorm_kernel<<<S, 256, 0, stream>>>(d_x, lw.ln_1_w, lw.ln_1_b, d_h);
        vis_gemm_bf16(h, d_h, lw.qkv_w, d_tmp, S, 3 * VIS_HIDDEN, VIS_HIDDEN);
        vis_add_bias_rope_qkv_kernel<<<S, 512, 0, stream>>>(d_tmp, lw.qkv_b, d_q, d_k, d_v, S, grid_cols);
        vis_flash_attn_kernel<<<dim3((S + 3) / 4, VIS_HEADS), 128, 0, stream>>>(d_q, d_k, d_v, d_attn, S);
        vis_gemm_bf16(h, d_attn, lw.out_proj_w, d_h, S, VIS_HIDDEN, VIS_HIDDEN);
        vis_bias_layerscale_res_kernel<<<x_blks, 256, 0, stream>>>(d_x, d_h, lw.out_proj_b, lw.ls_1_gamma, x_elems);

        // MLP block
        vis_layernorm_kernel<<<S, 256, 0, stream>>>(d_x, lw.ln_2_w, lw.ln_2_b, d_h);
        vis_gemm_bf16(h, d_h, lw.c_fc_w, d_tmp, S, VIS_INTERMEDIATE, VIS_HIDDEN);
        vis_bias_gelu_kernel<<<mlp_blks, 256, 0, stream>>>(d_tmp, lw.c_fc_b, VIS_INTERMEDIATE, mlp_elems);
        vis_gemm_bf16(h, d_tmp, lw.c_proj_w, d_h, S, VIS_HIDDEN, VIS_INTERMEDIATE);
        vis_bias_layerscale_res_kernel<<<x_blks, 256, 0, stream>>>(d_x, d_h, lw.c_proj_b, lw.ls_2_gamma, x_elems);
    }

    // 5. Spatial merger (drop CLS; X[1..N_patch] is already [n_tok, 4096] in merge-block order!) + 2-layer GELU projector
    vis_gemm_bf16(h, d_x + VIS_HIDDEN, vw.proj_in, d_tmp, n_tok, VIS_PROJ_HIDDEN, VIS_MERGE_SIZE * VIS_MERGE_SIZE * VIS_HIDDEN);
    const int proj_elems = n_tok * VIS_PROJ_HIDDEN;
    vis_bias_gelu_kernel<<<(proj_elems + 255) / 256, 256, 0, stream>>>(d_tmp, nullptr, VIS_PROJ_HIDDEN, proj_elems);
    vis_gemm_bf16(h, d_tmp, vw.proj_out, d_attn, n_tok, HIDDEN, VIS_PROJ_HIDDEN);

    // 6. Convert BF16 -> FP32 [n_tok, HIDDEN]
    const int out_elems = n_tok * HIDDEN;
    vis_bf16_to_f32_kernel<<<(out_elems + 255) / 256, 256, 0, stream>>>(d_attn, d_out_embeds, out_elems);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(ws));
}

void launch_scatter_vision_embeds(
    float* hidden,
    const float* vis_embeds,
    const int32_t* d_dst_rows,
    const int32_t* d_src_rows,
    int n_slots,
    cudaStream_t stream) {
    if (n_slots <= 0) return;
    scatter_vision_embeds_kernel<<<n_slots, 256, 0, stream>>>(hidden, vis_embeds, d_dst_rows, d_src_rows, n_slots);
}

} // namespace squid
