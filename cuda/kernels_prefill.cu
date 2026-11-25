// Fast Batched Prefill Kernels (B <= 256):
// 1. Coalesced NVFP4/FP8/BF16 -> FP16 dequant + cuBLAS FP16 Tensor Core GEMM for dense, router & controllers.
// 2. Blackwell mma.sync.aligned.m16n8k16 FP16 Tensor Core Multi-Token MoE (w13 softcapped SwiGLU + w2 zero-atomic).
// 3. 1-launch batched B<=256 causal sconv + MoE slot reduction + residual + depth-attn controller.

#include "config.hpp"
#include "kernels.hpp"
#include "kernels_common.cuh"
#include "weights.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace squid {

namespace {

static cublasHandle_t g_cublas = nullptr;

static cublasHandle_t get_cublas(cudaStream_t stream) {
    if (!g_cublas) {
        cublasStatus_t st = cublasCreate(&g_cublas);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasCreate failed: %d\n", (int)st);
            std::exit(1);
        }
        cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH);
    }
    cublasSetStream(g_cublas, stream);
    return g_cublas;
}

__global__ void p_f32_to_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst, size_t n) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(src + i);
        __half2 h0 = __floats2half2_rn(v.x, v.y);
        __half2 h1 = __floats2half2_rn(v.z, v.w);
        *reinterpret_cast<__half2*>(dst + i) = h0;
        *reinterpret_cast<__half2*>(dst + i + 2) = h1;
    } else {
        for (size_t k = i; k < n && k < i + 4; ++k) dst[k] = __float2half(src[k]);
    }
}

__global__ void p_dequant_nvfp4_to_f16_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    __half* __restrict__ out_f16,
    size_t total_blocks32) {
    size_t b32 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b32 >= total_blocks32) return;
    uint4 w16 = ld_stream_u4(reinterpret_cast<const uint4*>(w_data) + b32);
    uint16_t s16 = ld_scale_u16(reinterpret_cast<const uint16_t*>(w_scales) + b32);
    __half2 sc2 = cvt_e4m3x2_to_half2(s16);
    __half2 slo = __low2half2(sc2);
    __half2 shi = __high2half2(sc2);

    __half2 h[8];
    cvt_e2m1x8_to_half2x4(w16.x, h[0], h[1], h[2], h[3]);
    cvt_e2m1x8_to_half2x4(w16.y, h[4], h[5], h[6], h[7]);
    #pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = __hmul2(h[i], slo);
    uint4* dst4 = reinterpret_cast<uint4*>(out_f16 + b32 * 32);
    dst4[0] = *reinterpret_cast<const uint4*>(&h[0]);
    dst4[1] = *reinterpret_cast<const uint4*>(&h[4]);

    cvt_e2m1x8_to_half2x4(w16.z, h[0], h[1], h[2], h[3]);
    cvt_e2m1x8_to_half2x4(w16.w, h[4], h[5], h[6], h[7]);
    #pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = __hmul2(h[i], shi);
    dst4[2] = *reinterpret_cast<const uint4*>(&h[0]);
    dst4[3] = *reinterpret_cast<const uint4*>(&h[4]);
}

__global__ void p_dequant_fp8_to_f16_kernel(
    const uint8_t* __restrict__ w_data,
    const float* __restrict__ row_scales,
    __half* __restrict__ out_f16,
    int N, int K) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)N * (K >> 1);
    if (idx >= total) return;
    int row = (int)(idx / (K >> 1));
    float sc = __ldg(row_scales + row);
    uint16_t raw2 = __ldg(reinterpret_cast<const uint16_t*>(w_data) + idx);
    float2 f = __half22float2(cvt_e4m3x2_to_half2(raw2));
    reinterpret_cast<__half2*>(out_f16)[idx] = __floats2half2_rn(f.x * sc, f.y * sc);
}

__global__ void p_dequant_bf16_to_f16_kernel(
    const __nv_bfloat16* __restrict__ w_data,
    __half* __restrict__ out_f16,
    size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out_f16[i] = __float2half(__bfloat162float(w_data[i]));
}

__device__ __forceinline__ float4 load_sconv_src4(
    const float* __restrict__ src_or_slots,
    int is_slotted,
    int b,
    int c4) {
    if (is_slotted) {
        const float* sb = src_or_slots + (size_t)b * TOP_K * HIDDEN + c4;
        float4 s = *reinterpret_cast<const float4*>(sb);
        #pragma unroll
        for (int k = 1; k < TOP_K; ++k) {
            float4 v = *reinterpret_cast<const float4*>(sb + k * HIDDEN);
            s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
        }
        return s;
    }
    return *reinterpret_cast<const float4*>(src_or_slots + (size_t)b * HIDDEN + c4);
}

// 2D-parallel float4-vectorized batched causal hidden-dim 4-tap FIR sconv + optional 8-slot MoE reduction + residual!
// Grid: dim3((HIDDEN/4 + 255)/256, (B+T_BLK-1)/T_BLK).
constexpr int SCONV_T_BLK = 32;

__global__ void p_sconv_post_res_kernel(
    const float* __restrict__ src_or_slots,
    int is_slotted,
    const __nv_bfloat16* __restrict__ sconv_w,
    float* __restrict__ ring,
    const float* __restrict__ residual,
    float* __restrict__ out_res,
    float* __restrict__ pre_tap_out, // optional (e.g. prelude)
    float* __restrict__ blocks,      // optional: commit partial into blocks[b, block_idx]
    const float* __restrict__ partial,
    int block_idx,
    int is_boundary,
    int write_conv_only,
    int update_ring,
    int B,
    const int32_t* __restrict__ d_pos) {
    int c4 = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    if (c4 >= HIDDEN) return;

    int b0 = blockIdx.y * SCONV_T_BLK;
    int b1 = min(b0 + SCONV_T_BLK, B);
    if (b0 >= B) return;

    int pos0 = *d_pos;
    float w0[4], w1[4], w2[4], w3[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __nv_bfloat16* wp = sconv_w + (c4 + i) * 4;
        w0[i] = __bfloat162float(wp[0]);
        w1[i] = __bfloat162float(wp[1]);
        w2[i] = __bfloat162float(wp[2]);
        w3[i] = __bfloat162float(wp[3]);
    }

    auto get_hist4 = [&](int b_idx) -> float4 {
        if (b_idx >= 0) return load_sconv_src4(src_or_slots, is_slotted, b_idx, c4);
        int p_hist = pos0 + b_idx;
        return (p_hist >= 0)
            ? *reinterpret_cast<const float4*>(ring + (p_hist & 15) * HIDDEN + c4)
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    };
    float4 h0 = get_hist4(b0 - 3);
    float4 h1 = get_hist4(b0 - 2);
    float4 h2 = get_hist4(b0 - 1);

    for (int b = b0; b < b1; ++b) {
        float4 cur = load_sconv_src4(src_or_slots, is_slotted, b, c4);
        if (update_ring && b >= B - 16) {
            *reinterpret_cast<float4*>(ring + ((pos0 + b) & 15) * HIDDEN + c4) = cur;
        }
        float4 conv = make_float4(
            (w0[0] * h0.x + w1[0] * h1.x + w2[0] * h2.x + w3[0] * cur.x + cur.x) * INV_SQRT_5,
            (w0[1] * h0.y + w1[1] * h1.y + w2[1] * h2.y + w3[1] * cur.y + cur.y) * INV_SQRT_5,
            (w0[2] * h0.z + w1[2] * h1.z + w2[2] * h2.z + w3[2] * cur.z + cur.z) * INV_SQRT_5,
            (w0[3] * h0.w + w1[3] * h1.w + w2[3] * h2.w + w3[3] * cur.w + cur.w) * INV_SQRT_5);
        h0 = h1; h1 = h2; h2 = cur;

        if (write_conv_only) {
            *reinterpret_cast<float4*>(out_res + (size_t)b * HIDDEN + c4) = conv;
        } else {
            float4 res_v;
            if (is_boundary) {
                if (blocks && partial) {
                    *reinterpret_cast<float4*>(blocks + ((size_t)b * 4 + block_idx) * HIDDEN + c4) =
                        *reinterpret_cast<const float4*>(partial + (size_t)b * HIDDEN + c4);
                }
                res_v = conv;
            } else {
                float4 r0 = *reinterpret_cast<const float4*>(residual + (size_t)b * HIDDEN + c4);
                res_v = make_float4(r0.x + conv.x, r0.y + conv.y, r0.z + conv.z, r0.w + conv.w);
            }
            *reinterpret_cast<float4*>(out_res + (size_t)b * HIDDEN + c4) = res_v;
            if (pre_tap_out) *reinterpret_cast<float4*>(pre_tap_out + (size_t)b * HIDDEN + c4) = res_v;
        }
    }
}

__global__ void p_sconv_ring_commit_kernel(
    const float* __restrict__ src_or_slots,
    int is_slotted,
    float* __restrict__ ring,
    int B,
    const int32_t* __restrict__ d_pos) {
    int c4 = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    if (c4 >= HIDDEN) return;
    int pos0 = *d_pos;
    int b0 = max(0, B - 16);
    for (int b = b0; b < B; ++b) {
        *reinterpret_cast<float4*>(ring + ((pos0 + b) & 15) * HIDDEN + c4) =
            load_sconv_src4(src_or_slots, is_slotted, b, c4);
    }
}

// 1-CTA-per-token (576 threads) runtime-gated directional ablation + residual/boundary commit for prefill.
__global__ void __launch_bounds__(576)
p_ablate_post_res_kernel(
    float* __restrict__ out_res,         // in: raw post-sconv y; out: ablated residual
    const float* __restrict__ residual,
    float* __restrict__ pre_tap_out,
    float* __restrict__ blocks,
    const float* __restrict__ partial,
    int block_idx,
    int is_boundary,
    const float* __restrict__ abl_r,
    const float* __restrict__ abl_d,
    const float* __restrict__ abl_p) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int c4 = tid << 2;

    __shared__ float sm_r[18];
    __shared__ float sm_d[18];
    __shared__ float s_coeff;

    float4 y = *reinterpret_cast<const float4*>(out_res + (size_t)b * HIDDEN + c4);
    float4 rv = *reinterpret_cast<const float4*>(abl_r + c4);
    float dr = warp_sum(y.x * rv.x + y.y * rv.y + y.z * rv.z + y.w * rv.w);
    if (lane == 0) sm_r[warp] = dr;
    if (abl_d) {
        float4 dv = *reinterpret_cast<const float4*>(abl_d + c4);
        float dd = warp_sum(y.x * dv.x + y.y * dv.y + y.z * dv.z + y.w * dv.w);
        if (lane == 0) sm_d[warp] = dd;
    }
    __syncthreads();
    if (warp == 0) {
        float tot_r = warp_sum((lane < 18) ? sm_r[lane] : 0.0f);
        float tot_d = abl_d ? warp_sum((lane < 18) ? sm_d[lane] : 0.0f) : 0.0f;
        if (lane == 0) {
            float4 ap = __ldg(reinterpret_cast<const float4*>(abl_p));
            s_coeff = (!abl_d || tot_d > ap.w) ? (ap.x * fmaxf(tot_r - ap.y, ap.z)) : 0.0f;
        }
    }
    __syncthreads();
    float k = s_coeff;
    y = make_float4(y.x - k * rv.x, y.y - k * rv.y, y.z - k * rv.z, y.w - k * rv.w);

    float4 res_v;
    if (is_boundary) {
        if (blocks && partial) {
            *reinterpret_cast<float4*>(blocks + ((size_t)b * 4 + block_idx) * HIDDEN + c4) =
                *reinterpret_cast<const float4*>(partial + (size_t)b * HIDDEN + c4);
        }
        res_v = y;
    } else {
        float4 base = *reinterpret_cast<const float4*>(residual + (size_t)b * HIDDEN + c4);
        res_v = make_float4(base.x + y.x, base.y + y.y, base.z + y.z, base.w + y.w);
    }
    *reinterpret_cast<float4*>(out_res + (size_t)b * HIDDEN + c4) = res_v;
    if (pre_tap_out) *reinterpret_cast<float4*>(pre_tap_out + (size_t)b * HIDDEN + c4) = res_v;
}

// Coalesced float4 copy into contiguous [4, B, HIDDEN] tap buffer for 1-shot batched DSpark prefill commit!
__global__ void p_copy_taps_chunked_kernel(
    const float* __restrict__ src, // [B, HIDDEN]
    float* __restrict__ dst_taps,  // [4, B, HIDDEN]
    int tap_slot,
    int B) {
    size_t idx4 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total4 = ((size_t)B * HIDDEN) >> 2;
    if (idx4 >= total4) return;
    reinterpret_cast<float4*>(dst_taps + (size_t)tap_slot * B * HIDDEN)[idx4] =
        reinterpret_cast<const float4*>(src)[idx4];
}

// Batched B<=2048 Controller Prep (576 threads = float4 exact, zero loops!):
// Computes msg = depth_attn(saved, prior_last, cp_q), cat_h = [rmsnorm(x), rmsnorm(msg)] in FP16!
__global__ void __launch_bounds__(576)
p_ctrl_prep_h_kernel(
    const float* __restrict__ x,
    const float* __restrict__ saved,
    const float* __restrict__ prior_last,
    const float* __restrict__ cp_q,
    float* __restrict__ out_msg,   // [B, HIDDEN]
    __half* __restrict__ out_cat_h // [B, 2*HIDDEN]
) {
    int b = blockIdx.x;
    int tid = threadIdx.x; // 0..575
    int lane = tid & 31;
    int warp = tid >> 5;   // 0..17
    int c4 = tid << 2;

    float4 sv = *reinterpret_cast<const float4*>(saved + (size_t)b * HIDDEN + c4);
    float4 pl = *reinterpret_cast<const float4*>(prior_last + (size_t)b * HIDDEN + c4);
    float4 xv = *reinterpret_cast<const float4*>(x + (size_t)b * HIDDEN + c4);
    float4 qv = *reinterpret_cast<const float4*>(cp_q + c4);

    float d0 = warp_sum(fmaf(qv.x, sv.x, fmaf(qv.y, sv.y, fmaf(qv.z, sv.z, qv.w * sv.w))));
    float d1 = warp_sum(fmaf(qv.x, pl.x, fmaf(qv.y, pl.y, fmaf(qv.z, pl.z, qv.w * pl.w))));
    float xsq = warp_sum(fmaf(xv.x, xv.x, fmaf(xv.y, xv.y, fmaf(xv.z, xv.z, xv.w * xv.w))));

    __shared__ float sm0[18], sm1[18], smx[18];
    if (lane == 0) { sm0[warp] = d0; sm1[warp] = d1; smx[warp] = xsq; }
    __syncthreads();

    __shared__ float w0, w1, inv_x;
    if (warp == 0) {
        float t0 = warp_sum((lane < 18) ? sm0[lane] : 0.0f);
        float t1 = warp_sum((lane < 18) ? sm1[lane] : 0.0f);
        float tx = warp_sum((lane < 18) ? smx[lane] : 0.0f);
        if (lane == 0) {
            float l0 = t0 * INV_SQRT_HIDDEN;
            float l1 = t1 * INV_SQRT_HIDDEN;
            float mx = fmaxf(l0, l1);
            float e0 = expf(l0 - mx), e1 = expf(l1 - mx);
            float inv = 1.0f / (e0 + e1);
            w0 = e0 * inv;
            w1 = e1 * inv;
            inv_x = rsqrtf(tx / float(HIDDEN) + RMS_EPS);
        }
    }
    __syncthreads();

    float ww0 = w0, ww1 = w1, ix = inv_x;
    float4 mv = make_float4(
        ww0 * sv.x + ww1 * pl.x,
        ww0 * sv.y + ww1 * pl.y,
        ww0 * sv.z + ww1 * pl.z,
        ww0 * sv.w + ww1 * pl.w);
    *reinterpret_cast<float4*>(out_msg + (size_t)b * HIDDEN + c4) = mv;

    __half* cat_b = out_cat_h + (size_t)b * (2 * HIDDEN);
    *reinterpret_cast<__half2*>(cat_b + c4)     = __floats2half2_rn(xv.x * ix, xv.y * ix);
    *reinterpret_cast<__half2*>(cat_b + c4 + 2) = __floats2half2_rn(xv.z * ix, xv.w * ix);

    float msq = warp_sum(fmaf(mv.x, mv.x, fmaf(mv.y, mv.y, fmaf(mv.z, mv.z, mv.w * mv.w))));
    if (lane == 0) sm0[warp] = msq;
    __syncthreads();
    __shared__ float inv_m;
    if (warp == 0) {
        float tm = warp_sum((lane < 18) ? sm0[lane] : 0.0f);
        if (lane == 0) inv_m = rsqrtf(tm / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();
    float im = inv_m;
    *reinterpret_cast<__half2*>(cat_b + HIDDEN + c4)     = __floats2half2_rn(mv.x * im, mv.y * im);
    *reinterpret_cast<__half2*>(cat_b + HIDDEN + c4 + 2) = __floats2half2_rn(mv.z * im, mv.w * im);
}

// Batched B<=2048 Controller Finish (576 threads = float4 exact, skips dead F32 out when out == nullptr!):
// merged = (1 - tanh(forget)) * x + tanh(gain) * msg -> RMSNorm * post_norm -> out_h (f16) [& optional out (f32)]
__global__ void __launch_bounds__(576)
p_ctrl_finish_kernel(
    const float* __restrict__ x,
    const float* __restrict__ msg,
    const float* __restrict__ forget,
    const float* __restrict__ gain,
    const __nv_bfloat16* __restrict__ post_norm,
    float* __restrict__ out,
    __half* __restrict__ out_h) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int c4 = tid << 2;

    float4 xv = *reinterpret_cast<const float4*>(x + (size_t)b * HIDDEN + c4);
    float4 mv = *reinterpret_cast<const float4*>(msg + (size_t)b * HIDDEN + c4);
    float4 fv = *reinterpret_cast<const float4*>(forget + (size_t)b * HIDDEN + c4);
    float4 gv = *reinterpret_cast<const float4*>(gain + (size_t)b * HIDDEN + c4);

    float4 v = make_float4(
        (1.0f - fast_tanh(fv.x)) * xv.x + fast_tanh(gv.x) * mv.x,
        (1.0f - fast_tanh(fv.y)) * xv.y + fast_tanh(gv.y) * mv.y,
        (1.0f - fast_tanh(fv.z)) * xv.z + fast_tanh(gv.z) * mv.z,
        (1.0f - fast_tanh(fv.w)) * xv.w + fast_tanh(gv.w) * mv.w);

    float sq = warp_sum(fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, v.w * v.w))));
    __shared__ float sm[18];
    if (lane == 0) sm[warp] = sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (warp == 0) {
        float tot = warp_sum((lane < 18) ? sm[lane] : 0.0f);
        if (lane == 0) inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();
    float ir = inv_rms;
    uint2 nw4 = *reinterpret_cast<const uint2*>(post_norm + c4);
    const __nv_bfloat162* nb = reinterpret_cast<const __nv_bfloat162*>(&nw4);
    float2 n0 = __bfloat1622float2(nb[0]);
    float2 n1 = __bfloat1622float2(nb[1]);
    float4 o = make_float4(v.x * ir * n0.x, v.y * ir * n0.y, v.z * ir * n1.x, v.w * ir * n1.y);
    if (out) *reinterpret_cast<float4*>(out + (size_t)b * HIDDEN + c4) = o;
    if (out_h) {
        __half* ohb = out_h + (size_t)b * HIDDEN + c4;
        *reinterpret_cast<__half2*>(ohb)     = __floats2half2_rn(o.x, o.y);
        *reinterpret_cast<__half2*>(ohb + 2) = __floats2half2_rn(o.z, o.w);
    }
}

// Batched B<=2048 DSpark Mix (576 threads float4): mixes contiguous taps[4, B, HIDDEN] -> mix_h[3, B, HIDDEN] + f_ring[last 16]!
__global__ void __launch_bounds__(576)
p_dspark_mix_h_kernel(
    const float* __restrict__ taps,       // [4, B, HIDDEN]
    int B,
    const float* __restrict__ fuse_alpha, // [4, 4]
    const __nv_bfloat16* __restrict__ n0,
    const __nv_bfloat16* __restrict__ n1,
    const __nv_bfloat16* __restrict__ n2,
    const __nv_bfloat16* __restrict__ n3,
    __half* __restrict__ out_mix_h,       // [3, B, HIDDEN]
    float* __restrict__ f_ring,           // [16, HIDDEN]
    const int32_t* __restrict__ d_pos) {
    int b = blockIdx.x;
    int m = blockIdx.y; // 0..3
    if (m == 3 && b < B - 16) return;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int c4 = tid << 2;

    const float a0 = fuse_alpha[m * 4 + 0];
    const float a1 = fuse_alpha[m * 4 + 1];
    const float a2 = fuse_alpha[m * 4 + 2];
    const float a3 = fuse_alpha[m * 4 + 3];

    float4 v0 = *reinterpret_cast<const float4*>(taps + ((size_t)0 * B + b) * HIDDEN + c4);
    float4 v1 = *reinterpret_cast<const float4*>(taps + ((size_t)1 * B + b) * HIDDEN + c4);
    float4 v2 = *reinterpret_cast<const float4*>(taps + ((size_t)2 * B + b) * HIDDEN + c4);
    float4 v3 = *reinterpret_cast<const float4*>(taps + ((size_t)3 * B + b) * HIDDEN + c4);

    float4 v = make_float4(
        a0 * v0.x + a1 * v1.x + a2 * v2.x + a3 * v3.x,
        a0 * v0.y + a1 * v1.y + a2 * v2.y + a3 * v3.y,
        a0 * v0.z + a1 * v1.z + a2 * v2.z + a3 * v3.z,
        a0 * v0.w + a1 * v1.w + a2 * v2.w + a3 * v3.w);

    float sq = warp_sum(fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, v.w * v.w))));
    __shared__ float sm[18];
    if (lane == 0) sm[warp] = sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (warp == 0) {
        float tot = warp_sum((lane < 18) ? sm[lane] : 0.0f);
        if (lane == 0) inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();

    const __nv_bfloat16* nw = (m == 0) ? n0 : (m == 1 ? n1 : (m == 2 ? n2 : n3));
    uint2 nw4 = *reinterpret_cast<const uint2*>(nw + c4);
    const __nv_bfloat162* nb = reinterpret_cast<const __nv_bfloat162*>(&nw4);
    float2 nf0 = __bfloat1622float2(nb[0]);
    float2 nf1 = __bfloat1622float2(nb[1]);
    float ir = inv_rms;
    float4 o = make_float4(v.x * ir * nf0.x, v.y * ir * nf0.y, v.z * ir * nf1.x, v.w * ir * nf1.y);
    if (m == 3) {
        float* dst = f_ring + (size_t)(((*d_pos) + b) & 15) * HIDDEN + c4;
        *reinterpret_cast<float4*>(dst) = o;
    } else {
        __half* dst_h = out_mix_h + ((size_t)m * B + b) * HIDDEN + c4;
        *reinterpret_cast<__half2*>(dst_h)     = __floats2half2_rn(o.x, o.y);
        *reinterpret_cast<__half2*>(dst_h + 2) = __floats2half2_rn(o.z, o.w);
    }
}

// Batched DSpark ctx KV store (with b_off = max(0, B - cap) so local ring writes are 100% race-free!)
template <int KV_FMT>
__global__ void p_dspark_store_ctx_kv_kernel(
    const float* __restrict__ kv,
    const __nv_bfloat16* __restrict__ k_norm,
    void* __restrict__ k_cache,
    void* __restrict__ v_cache,
    __half* __restrict__ k_scale,
    __half* __restrict__ v_scale,
    int cap,
    int b_off,
    const int32_t* __restrict__ d_pos) {
    int b = b_off + blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int pos = (*d_pos) + b;
    int slot = pos % cap;

    const float* kp = kv + (size_t)b * (2 * KV_DIM) + h * HEAD_DIM;
    const float* vp = kv + (size_t)b * (2 * KV_DIM) + KV_DIM + h * HEAD_DIM;

    __shared__ float sm[4];
    store_kv_head_128<KV_FMT>(vp[tid], slot, h, tid, v_cache, v_scale, sm);
    __syncthreads();

    float kv_val = kp[tid];
    float sq = warp_sum(kv_val * kv_val);
    if ((tid & 31) == 0) sm[tid >> 5] = sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (tid == 0) inv_rms = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / float(HEAD_DIM) + RMS_EPS);
    __syncthreads();

    __shared__ float kn[HEAD_DIM];
    kn[tid] = kv_val * inv_rms * __bfloat162float(k_norm[tid]);
    __syncthreads();

    if (tid < 64) {
        float freq = 1.0f / powf(MTP_ROPE_THETA, float(2 * tid) / float(HEAD_DIM));
        float ang = float(pos) * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = kn[tid], x1 = kn[tid + 64];
        kn[tid]      = x0 * c - x1 * s;
        kn[tid + 64] = x0 * s + x1 * c;
    }
    __syncthreads();
    store_kv_head_128<KV_FMT>(kn[tid], slot, h, tid, k_cache, k_scale, sm);
}

// Per-token top-8 selection (1 warp per token, grid = B) — identical tie-breaking to kernels_inkling.cu!
__global__ void p_route_topk_only_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ router_bias,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_weights) {
    int b = blockIdx.x;
    int lane = threadIdx.x;

    float sc[4], bi[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int e = lane + 32 * i;
        float raw = fast_sigmoid(logits[(size_t)b * NUM_EXPERTS + e]);
        sc[i] = raw;
        bi[i] = raw + router_bias[e];
    }

    int winners[TOP_K];
    float chosen[TOP_K];
    for (int k = 0; k < TOP_K; ++k) {
        float best = -1e30f;
        int best_e = NUM_EXPERTS;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int e = lane + 32 * i;
            if (bi[i] > best || (bi[i] == best && e < best_e)) {
                best = bi[i];
                best_e = e;
            }
        }
        float top = warp_max(best);
        int cand = (best == top) ? best_e : NUM_EXPERTS;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            cand = min(cand, __shfl_xor_sync(0xffffffffu, cand, off));
        }
        cand = min(cand, NUM_EXPERTS - 1);
        float sel = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (lane + 32 * i == cand) {
                sel = sc[i];
                bi[i] = -1e30f;
            }
        }
        chosen[k] = warp_sum(sel);
        winners[k] = cand;
    }

    if (lane == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) total += chosen[k];
        float inv = ROUTE_SCALE / (total + 1e-20f);
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            out_ids[b * TOP_K + k] = winners[k];
            out_weights[b * TOP_K + k] = chosen[k] * inv;
        }
    }
}

__global__ void p_sort_experts_kernel(
    const int32_t* __restrict__ route_ids,
    const float* __restrict__ route_w,
    int B,
    int TR,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ sorted_tok,
    int32_t* __restrict__ sorted_slot,
    float* __restrict__ sorted_w) {
    __shared__ int32_t counts[NUM_EXPERTS];
    __shared__ int32_t cursors[NUM_EXPERTS];
    int tid = threadIdx.x;
    counts[tid] = 0;
    __syncthreads();

    int total = B * TOP_K;
    for (int idx = tid; idx < total; idx += NUM_EXPERTS) {
        atomicAdd(&counts[route_ids[idx]], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int sum = 0;
        int nt = 0;
        int32_t* tile_exp = expert_offsets + 130;
        int32_t* tile_st  = expert_offsets + 386;
        int32_t* tile_en  = expert_offsets + 642;
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            expert_offsets[e] = sum;
            cursors[e] = sum;
            int cnt = counts[e];
            for (int off = 0; off < cnt; off += TR) {
                tile_exp[nt] = e;
                tile_st[nt]  = sum + off;
                tile_en[nt]  = sum + min(off + TR, cnt);
                nt++;
            }
            sum += cnt;
        }
        expert_offsets[NUM_EXPERTS] = sum;
        expert_offsets[129] = nt;
    }
    __syncthreads();

    for (int idx = tid; idx < total; idx += NUM_EXPERTS) {
        int e = route_ids[idx];
        int dst = atomicAdd(&cursors[e], 1);
        sorted_tok[dst] = idx >> 3;
        sorted_slot[dst] = idx & 7;
        sorted_w[dst] = route_w[idx];
    }
}

// ============================================================================
// Multi-Stage cp.async 128x128x64 Blackwell FP16 TC Grouped MoE GEMM (Prefill)
// Adapted from satellitedown/inkling-2-mini-abliterated-ingine wgemm.cu (Apache-2.0)
// and specialized for Blackwell sm_121a hardware cvt.rn.f16x2.e2m1x2 + FP16 MMA!
// ============================================================================

enum class WgemmMode { Up, Down };
constexpr int WG_TN = 128;
constexpr int WG_BK = 64;
constexpr int WG_AS = 3;
constexpr int WG_NT = 256;
constexpr int WG_KCH = WG_BK / 8; // 8 16-byte chunks per activation row

__device__ __forceinline__ uint32_t wg_smem_addr(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void wg_cp_async16(void* smem_dst, const void* gmem_src, int src_bytes = 16) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(wg_smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
}

__device__ __forceinline__ void wg_cp_async4(void* smem_dst, const void* gmem_src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :: "r"(wg_smem_addr(smem_dst)), "l"(gmem_src));
}

__device__ __forceinline__ void wg_cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int Groups>
__device__ __forceinline__ void wg_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(Groups));
}

__device__ __forceinline__ void wg_ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void wg_mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int Chunks>
__device__ __forceinline__ int wg_swz(int row, int chunk) {
    return row * (Chunks * 8) + ((chunk ^ (row & 7)) << 3);
}

template <int FMT, int K, int TM>
struct WgCfg {
    static constexpr int CODE_ROW  = (FMT == 0) ? (K / 2) : K;
    static constexpr int SCALE_ROW = (FMT == 0) ? (K / 16) : (K / 32);
    static constexpr int WK        = (FMT == 0) ? 128 : 64;
    static constexpr int R         = WK / WG_BK;
    static constexpr int KC        = K / WK;
    static constexpr int WS        = (TM <= 32) ? ((FMT == 0) ? 6 : 7) : ((FMT == 0) ? 3 : 4);
    static_assert(R * (WS - 1) >= WG_AS);
    static constexpr int WCODE     = (FMT == 0) ? (WK / 2) : WK;
    static constexpr int WSTRIDE   = WCODE + 16;
    static constexpr int W_BYTES   = WG_TN * WSTRIDE;
    static constexpr int A_BYTES   = TM * WG_BK * 2;
    static constexpr int SC_BYTES  = WG_TN * SCALE_ROW;
    static constexpr int OFF_W     = WG_AS * A_BYTES;
    static constexpr int OFF_SC    = OFF_W + WS * W_BYTES;
    static constexpr int SMEM      = (OFF_SC + SC_BYTES + 15) / 16 * 16;
    static_assert(K % WK == 0 && SMEM <= 101376);
};

template <int MT, int TM, class After>
__device__ __forceinline__ void wg_mma_step(
    float (&acc)[TM / 16][2][4], const uint16_t* A,
    const uint32_t (&b)[WG_BK / 16][2][2], int lane, After&& after) {
    #pragma unroll
    for (int kk = 0; kk < WG_BK / 16; ++kk) {
        uint32_t a[MT][4];
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            wg_ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                           wg_smem_addr(A + wg_swz<WG_KCH>(mt * 16 + (lane & 15), kk * 2 + (lane >> 4))));
        }
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            wg_mma_f16(acc[mt][0], a[mt], b[kk][0][0], b[kk][0][1]);
            wg_mma_f16(acc[mt][1], a[mt], b[kk][1][0], b[kk][1][1]);
        }
        after(kk);
    }
}

template <int FMT, WgemmMode MODE, int K, int TM>
__global__ void __launch_bounds__(WG_NT, 1)
p_wgemm_kernel(
    const uint8_t* __restrict__ w_codes,
    const uint8_t* __restrict__ w_scales,
    const float* __restrict__ w_factors,
    const __half* __restrict__ x_h,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ sorted_tok,
    const int32_t* __restrict__ sorted_slot,
    const float* __restrict__ sorted_w,
    __half* __restrict__ out_act_h,
    float* __restrict__ out_slots) {
    constexpr int KS = K / WG_BK;
    constexpr int NE = (MODE == WgemmMode::Up) ? (2 * INTERMEDIATE) : HIDDEN;
    constexpr int NTILES_N = NE / WG_TN;
    using C = WgCfg<FMT, K, TM>;

    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* a_smem = smem;
    unsigned char* w_smem = smem + C::OFF_W;
    uint8_t* sc_smem = smem + C::OFF_SC;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int tile_idx = blockIdx.x / NTILES_N;
    if (tile_idx >= expert_offsets[129]) return;

    const int e  = expert_offsets[130 + tile_idx];
    const int pb = expert_offsets[386 + tile_idx];
    const int en = expert_offsets[642 + tile_idx];
    const int n  = en - pb;
    const int n0 = (blockIdx.x % NTILES_N) * WG_TN;

    const uint8_t* code_tile  = w_codes  + ((size_t)e * NE + n0) * C::CODE_ROW;
    const uint8_t* scale_tile = w_scales + ((size_t)e * NE + n0) * C::SCALE_ROW;
    const uint16_t* a_base    = reinterpret_cast<const uint16_t*>(x_h);

    const uint16_t* arow[TM / 32];
    #pragma unroll
    for (int i = 0; i < TM / 32; ++i) {
        const int r = (tid >> 3) + 32 * i;
        const int rr = (r < n) ? r : 0;
        if constexpr (MODE == WgemmMode::Up) {
            arow[i] = a_base + (size_t)sorted_tok[pb + rr] * K;
        } else {
            arow[i] = a_base + (size_t)(pb + rr) * K;
        }
    }

    auto load_a = [&](int ks) {
        uint16_t* A = reinterpret_cast<uint16_t*>(a_smem + (ks % WG_AS) * C::A_BYTES);
        #pragma unroll
        for (int i = 0; i < TM / 32; ++i) {
            const int r = (tid >> 3) + 32 * i;
            const int c = tid & 7;
            wg_cp_async16(A + wg_swz<WG_KCH>(r, c), arow[i] + ks * WG_BK + c * 8, (r < n) ? 16 : 0);
        }
    };
    auto load_w = [&](int chunk) {
        unsigned char* ws = w_smem + (chunk % C::WS) * C::W_BYTES;
        constexpr int CPR = C::WCODE / 16;
        #pragma unroll
        for (int i = 0; i < WG_TN * CPR / WG_NT; ++i) {
            const int id = tid + i * WG_NT;
            const int row = id / CPR;
            const int ch = id % CPR;
            wg_cp_async16(ws + row * C::WSTRIDE + ch * 16,
                          code_tile + (size_t)row * C::CODE_ROW + chunk * C::WCODE + ch * 16);
        }
    };
    auto issue = [&](int j) {
        if (j < KS) load_a(j);
        wg_cp_commit();
    };

    for (int id = tid; id < C::SC_BYTES / 4; id += WG_NT) {
        int byte_off = id * 4;
        wg_cp_async4(sc_smem + byte_off, scale_tile + (size_t)(byte_off / C::SCALE_ROW) * C::SCALE_ROW + (byte_off % C::SCALE_ROW));
    }
    #pragma unroll
    for (int j = 0; j < C::WS - 1; ++j) {
        if (j < C::KC) load_w(j);
    }
    #pragma unroll
    for (int j = 0; j < WG_AS - 1; ++j) issue(j);

    const int c = lane & 3;
    const float factor = (FMT == 0) ? __ldg(w_factors + e) : 1.0f;
    const int mact = min((n + 15) / 16, TM / 16);
    float acc[TM / 16][2][4];
    #pragma unroll
    for (int a = 0; a < TM / 16; ++a) {
        #pragma unroll
        for (int b = 0; b < 2; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.0f;
    }

    auto scales_of = [&](int ks, __half2 (&sc)[2][4]) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int row = warp * 16 + nt * 8 + (lane >> 2);
            if constexpr (FMT == 0) {
                const uint32_t s4 = *reinterpret_cast<const uint32_t*>(sc_smem + row * C::SCALE_ROW + ks * 4);
                const __half2 lo = cvt_e4m3x2_to_half2(static_cast<uint16_t>(s4 & 0xFFFFu));
                const __half2 hi = cvt_e4m3x2_to_half2(static_cast<uint16_t>(s4 >> 16));
                sc[nt][0] = __low2half2(lo);
                sc[nt][1] = __high2half2(lo);
                sc[nt][2] = __low2half2(hi);
                sc[nt][3] = __high2half2(hi);
            } else {
                // Store raw UE8M0 bytes in sc_u8 via sc; decode in FP32 to avoid FP16 underflow/Inf*0=NaN!
                const uint8_t* ms = sc_smem + row * C::SCALE_ROW + ks * 2;
                const uint32_t u0 = (uint32_t)ms[0];
                const uint32_t u1 = (uint32_t)ms[1];
                sc[nt][0] = *reinterpret_cast<const __half2*>(&u0);
                sc[nt][1] = sc[nt][0];
                sc[nt][2] = *reinterpret_cast<const __half2*>(&u1);
                sc[nt][3] = sc[nt][2];
            }
        }
    };

    auto decode = [&](int ks, int kk, const __half2 (&sc)[2][4], uint32_t (&b)[WG_BK / 16][2][2]) {
        const unsigned char* ws = w_smem + ((ks / C::R) % C::WS) * C::W_BYTES;
        const int kc = (ks % C::R) * WG_BK + kk * 16;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int row = warp * 16 + nt * 8 + (lane >> 2);
            const __half2 s = sc[nt][kk];
            if constexpr (FMT == 0) {
                const uint2 wd = *reinterpret_cast<const uint2*>(ws + row * C::WSTRIDE + (kc >> 1));
                const uint32_t b0 = (wd.x >> (8 * c)) & 0xFFu;
                const uint32_t b1 = (wd.y >> (8 * c)) & 0xFFu;
                __half2 h0 = __hmul2(cvt_e2m1x2_to_half2(b0), s);
                __half2 h1 = __hmul2(cvt_e2m1x2_to_half2(b1), s);
                b[kk][nt][0] = *reinterpret_cast<const uint32_t*>(&h0);
                b[kk][nt][1] = *reinterpret_cast<const uint32_t*>(&h1);
            } else {
                const uint4 q = *reinterpret_cast<const uint4*>(ws + row * C::WSTRIDE + kc);
                const uint32_t w0 = (c >> 1) ? q.y : q.x;
                const uint32_t w1 = (c >> 1) ? q.w : q.z;
                const float sf = cvt_ue8m0_to_float(static_cast<uint8_t>(*reinterpret_cast<const uint32_t*>(&s)));
                float2 f0 = __half22float2(cvt_e4m3x2_to_half2(static_cast<uint16_t>(w0 >> (16 * (c & 1)))));
                float2 f1 = __half22float2(cvt_e4m3x2_to_half2(static_cast<uint16_t>(w1 >> (16 * (c & 1)))));
                __half2 h0 = __floats2half2_rn(f0.x * sf, f0.y * sf);
                __half2 h1 = __floats2half2_rn(f1.x * sf, f1.y * sf);
                b[kk][nt][0] = *reinterpret_cast<const uint32_t*>(&h0);
                b[kk][nt][1] = *reinterpret_cast<const uint32_t*>(&h1);
            }
            asm volatile("" : "+r"(b[kk][nt][0]), "+r"(b[kk][nt][1]));
        }
    };

    uint32_t bcur[WG_BK / 16][2][2], bnext[WG_BK / 16][2][2];
    #pragma unroll 1
    for (int ks = 0; ks < KS; ++ks) {
        wg_cp_wait<WG_AS - 2>();
        __syncthreads();
        const int chunk = ks / C::R;
        if (ks % C::R == 0 && chunk + C::WS - 1 < C::KC) load_w(chunk + C::WS - 1);
        issue(ks + WG_AS - 1);
        if (ks == 0) {
            __half2 sc0[2][4];
            scales_of(0, sc0);
            #pragma unroll
            for (int kk = 0; kk < WG_BK / 16; ++kk) decode(0, kk, sc0, bcur);
        }
        const uint16_t* A = reinterpret_cast<const uint16_t*>(a_smem + (ks % WG_AS) * C::A_BYTES);
        const bool more = (ks + 1 < KS);
        __half2 scn[2][4];
        if (more) scales_of(ks + 1, scn);
        auto after = [&](int kk) {
            if (more) decode(ks + 1, kk, scn, bnext);
        };
        switch (mact) {
            case 1: if constexpr (1 <= TM / 16) wg_mma_step<1, TM>(acc, A, bcur, lane, after); break;
            case 2: if constexpr (2 <= TM / 16) wg_mma_step<2, TM>(acc, A, bcur, lane, after); break;
            case 3: if constexpr (3 <= TM / 16) wg_mma_step<3, TM>(acc, A, bcur, lane, after); break;
            case 4: if constexpr (4 <= TM / 16) wg_mma_step<4, TM>(acc, A, bcur, lane, after); break;
            case 5: if constexpr (5 <= TM / 16) wg_mma_step<5, TM>(acc, A, bcur, lane, after); break;
            case 6: if constexpr (6 <= TM / 16) wg_mma_step<6, TM>(acc, A, bcur, lane, after); break;
            case 7: if constexpr (7 <= TM / 16) wg_mma_step<7, TM>(acc, A, bcur, lane, after); break;
            default: wg_mma_step<TM / 16, TM>(acc, A, bcur, lane, after); break;
        }
        #pragma unroll
        for (int kk = 0; kk < WG_BK / 16; ++kk) {
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                bcur[kk][nt][0] = bnext[kk][nt][0];
                bcur[kk][nt][1] = bnext[kk][nt][1];
            }
        }
    }
    wg_cp_wait<0>();

    #pragma unroll
    for (int mt = 0; mt < TM / 16; ++mt) {
        if (mt >= mact) continue;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = mt * 16 + (lane >> 2) + half * 8;
            if (row >= n) continue;
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int cl = warp * 16 + nt * 8 + c * 2;
                const float c0 = acc[mt][nt][2 * half] * factor;
                const float c1 = acc[mt][nt][2 * half + 1] * factor;
                if constexpr (MODE == WgemmMode::Up) {
                    const float g_cap = fast_tanh(c0 * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
                    const float u_cap = fast_tanh(c1 * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
                    out_act_h[(size_t)(pb + row) * INTERMEDIATE + (n0 + cl) / 2] =
                        __float2half((g_cap * fast_sigmoid(c0)) * u_cap);
                } else {
                    const int tok = sorted_tok[pb + row];
                    const float rw = sorted_w[pb + row];
                    atomicAdd(&out_slots[(size_t)tok * HIDDEN + n0 + cl + 0], c0 * rw);
                    atomicAdd(&out_slots[(size_t)tok * HIDDEN + n0 + cl + 1], c1 * rw);
                }
            }
        }
    }
}

template <int FMT, WgemmMode MODE, int K, int TM>
static void dispatch_wgemm_tm(
    const ExpertBank& eb, const __half* x_h,
    const int32_t* expert_offsets, const int32_t* sorted_tok, const int32_t* sorted_slot, const float* sorted_w,
    __half* out_act_h, float* out_slots, cudaStream_t stream) {
    constexpr int NE = (MODE == WgemmMode::Up) ? (2 * INTERMEDIATE) : HIDDEN;
    constexpr int NTILES_N = NE / WG_TN;
    constexpr int SMEM = WgCfg<FMT, K, TM>::SMEM;
    static const cudaError_t attr = cudaFuncSetAttribute(
        p_wgemm_kernel<FMT, MODE, K, TM>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    (void)attr;
    // Max 256 active tiles by pigeonhole bound (B<=512 w/ TR=32 or B<=2048 w/ TR=128)
    int grid = 256 * NTILES_N;
    p_wgemm_kernel<FMT, MODE, K, TM><<<grid, WG_NT, SMEM, stream>>>(
        eb.data, eb.scales, (FMT == 0) ? eb.expert_scale : nullptr,
        x_h, expert_offsets, sorted_tok, sorted_slot, sorted_w, out_act_h, out_slots);
}

// Race-free ring commit for the last min(B, 16) tokens of K and V projections!
__global__ void p_kv_ring_commit_kernel(
    const float* __restrict__ k_proj,
    const float* __restrict__ v_proj,
    float* __restrict__ k_ring,
    float* __restrict__ v_ring,
    int B,
    const int32_t* __restrict__ d_pos) {
    int c4 = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    if (c4 >= KV_DIM) return;
    int pos0 = *d_pos;
    int b0 = max(0, B - 16);
    float* dst_ring = blockIdx.y ? v_ring : k_ring;
    const float* src = blockIdx.y ? v_proj : k_proj;
    for (int b = b0; b < B; ++b) {
        *reinterpret_cast<float4*>(dst_ring + (size_t)((pos0 + b) & 15) * KV_DIM + c4) =
            *reinterpret_cast<const float4*>(src + (size_t)b * KV_DIM + c4);
    }
}

// 1-launch 2D-parallel fused local static sconv (read-only on ring!) + Q/R/K clip + V RMSNorm + BF16 KV store!
// Replaces 2x 4-CTA serial static_sconv_kernel + attn_prep_store_kernel (eliminates 141.5ms + 57.2ms!).
// Grid = dim3(64, B), Block = 128
__global__ void __launch_bounds__(128)
p_local_sconv_prep_store_kernel(
    float* __restrict__ qrgs,
    const float* __restrict__ k_proj,
    const float* __restrict__ v_proj,
    const __nv_bfloat16* __restrict__ k_sconv_w,
    const __nv_bfloat16* __restrict__ v_sconv_w,
    const float* __restrict__ k_ring,
    const float* __restrict__ v_ring,
    const __nv_bfloat16* __restrict__ q_clip_scale,
    const __nv_bfloat16* __restrict__ rel_clip_scale,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int kv_cap,
    const int32_t* __restrict__ d_pos) {
    int kind_id = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int p = (*d_pos) + b;
    constexpr int QRGS_STRIDE = Q_DIM + R_DIM + Q_DIM + NUM_HEADS;
    __shared__ float sm[4];
    __shared__ float factor;

    if (kind_id < 24) {
        int h = kind_id;
        float* qp = qrgs + (size_t)b * QRGS_STRIDE + h * HEAD_DIM;
        float val = qp[tid] * __bfloat162float(q_clip_scale[h * HEAD_DIM + tid]);
        float sq = warp_sum(val * val);
        if (lane == 0) sm[warp] = sq;
        __syncthreads();
        if (tid == 0) factor = fminf(QK_CLIP / (sqrtf(sm[0] + sm[1] + sm[2] + sm[3]) + 1e-6f), 1.0f);
        __syncthreads();
        qp[tid] = val * factor;
    } else if (kind_id < 48) {
        if (tid >= D_REL) return;
        int h = kind_id - 24;
        float* rp = qrgs + (size_t)b * QRGS_STRIDE + Q_DIM + h * D_REL;
        float val = rp[tid] * __bfloat162float(rel_clip_scale[h * D_REL + tid]);
        float sq = val * val;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) sq += __shfl_xor_sync(0xFFFFu, sq, off);
        rp[tid] = val * fminf(REL_CLIP / (sqrtf(sq) + 1e-6f), 1.0f);
    } else if (kind_id < 56) {
        int h = kind_id - 48;
        int c = h * HEAD_DIM + tid;
        float cur = k_proj[(size_t)b * KV_DIM + c];
        float t2 = (b >= 1) ? k_proj[(size_t)(b - 1) * KV_DIM + c] : ((p >= 1) ? k_ring[((p - 1) & 15) * KV_DIM + c] : 0.0f);
        float t1 = (b >= 2) ? k_proj[(size_t)(b - 2) * KV_DIM + c] : ((p >= 2) ? k_ring[((p - 2) & 15) * KV_DIM + c] : 0.0f);
        float t0 = (b >= 3) ? k_proj[(size_t)(b - 3) * KV_DIM + c] : ((p >= 3) ? k_ring[((p - 3) & 15) * KV_DIM + c] : 0.0f);
        uint2 w4 = __ldg(reinterpret_cast<const uint2*>(k_sconv_w + c * 4));
        const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&w4);
        float2 wa = __bfloat1622float2(wb[0]), wbb = __bfloat1622float2(wb[1]);
        float val = (fmaf(wa.x, t0, fmaf(wa.y, t1, fmaf(wbb.x, t2, wbb.y * cur))) + cur) * INV_SQRT_5;
        float sq = warp_sum(val * val);
        if (lane == 0) sm[warp] = sq;
        __syncthreads();
        if (tid == 0) factor = fminf(QK_CLIP / (sqrtf(sm[0] + sm[1] + sm[2] + sm[3]) + 1e-6f), 1.0f);
        __syncthreads();
        k_cache[(size_t)(p % kv_cap) * KV_DIM + c] = __float2bfloat16(val * factor);
    } else {
        int h = kind_id - 56;
        int c = h * HEAD_DIM + tid;
        float cur = v_proj[(size_t)b * KV_DIM + c];
        float t2 = (b >= 1) ? v_proj[(size_t)(b - 1) * KV_DIM + c] : ((p >= 1) ? v_ring[((p - 1) & 15) * KV_DIM + c] : 0.0f);
        float t1 = (b >= 2) ? v_proj[(size_t)(b - 2) * KV_DIM + c] : ((p >= 2) ? v_ring[((p - 2) & 15) * KV_DIM + c] : 0.0f);
        float t0 = (b >= 3) ? v_proj[(size_t)(b - 3) * KV_DIM + c] : ((p >= 3) ? v_ring[((p - 3) & 15) * KV_DIM + c] : 0.0f);
        uint2 w4 = __ldg(reinterpret_cast<const uint2*>(v_sconv_w + c * 4));
        const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&w4);
        float2 wa = __bfloat1622float2(wb[0]), wbb = __bfloat1622float2(wb[1]);
        float val = (fmaf(wa.x, t0, fmaf(wa.y, t1, fmaf(wbb.x, t2, wbb.y * cur))) + cur) * INV_SQRT_5;
        float sq = warp_sum(val * val);
        if (lane == 0) sm[warp] = sq;
        __syncthreads();
        if (tid == 0) factor = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / float(HEAD_DIM) + RMS_EPS);
        __syncthreads();
        v_cache[(size_t)(p % kv_cap) * KV_DIM + c] = __float2bfloat16(val * factor);
    }
}

// 1-launch 2D-parallel fused global dynamic sconv (read-only on ring!) + Q/R/K clip (log-pos tau) + V RMSNorm + KV store!
// Eliminates serial 8-CTA p_dynamic_sconv_kv_inplace_kernel (27.8ms -> 0ms!).
// Grid = dim3(64, B), Block = 128
template <int KV_FMT>
__global__ void __launch_bounds__(128)
p_global_attn_prep_store_kernel(
    float* __restrict__ qrgs,
    const float* __restrict__ k_proj,
    const float* __restrict__ v_proj,
    const __nv_bfloat16* __restrict__ k_sw,
    const __nv_bfloat16* __restrict__ v_sw,
    const float* __restrict__ k_taps,
    const float* __restrict__ v_taps,
    const float* __restrict__ k_ring,
    const float* __restrict__ v_ring,
    const __nv_bfloat16* __restrict__ q_clip_scale,
    const __nv_bfloat16* __restrict__ rel_clip_scale,
    void* __restrict__ k_cache,
    void* __restrict__ v_cache,
    __half* __restrict__ k_scale,
    __half* __restrict__ v_scale,
    int kv_cap,
    const int32_t* __restrict__ d_pos) {
    int kind_id = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int p = (*d_pos) + b;
    float ratio = fmaxf(float(p + 1) / float(LOG_SCALE_FLOOR), 1.0f);
    float tau = 1.0f + LOG_SCALE_ALPHA * logf(ratio);
    constexpr int QRGS_STRIDE = Q_DIM + R_DIM + Q_DIM + NUM_HEADS;
    __shared__ float sm[4];
    __shared__ float factor;

    if (kind_id < 24) {
        int h = kind_id;
        float* qp = qrgs + (size_t)b * QRGS_STRIDE + h * HEAD_DIM;
        float val = qp[tid] * __bfloat162float(q_clip_scale[h * HEAD_DIM + tid]);
        float sq = warp_sum(val * val);
        if (lane == 0) sm[warp] = sq;
        __syncthreads();
        if (tid == 0) factor = fminf(QK_CLIP / (sqrtf(sm[0] + sm[1] + sm[2] + sm[3]) + 1e-6f), 1.0f) * tau;
        __syncthreads();
        qp[tid] = val * factor;
    } else if (kind_id < 48) {
        if (tid >= D_REL) return;
        int h = kind_id - 24;
        float* rp = qrgs + (size_t)b * QRGS_STRIDE + Q_DIM + h * D_REL;
        float val = rp[tid] * __bfloat162float(rel_clip_scale[h * D_REL + tid]);
        float sq = val * val;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) sq += __shfl_xor_sync(0xFFFFu, sq, off);
        rp[tid] = val * fminf(REL_CLIP / (sqrtf(sq) + 1e-6f), 1.0f) * tau;
    } else if (kind_id < 56) {
        int h = kind_id - 48;
        int c = h * HEAD_DIM + tid;
        float cur = k_proj[(size_t)b * KV_DIM + c];
        float h1 = (b >= 1) ? k_proj[(size_t)(b - 1) * KV_DIM + c] : ((p >= 1) ? k_ring[((p - 1) & 15) * KV_DIM + c] : 0.0f);
        float h2 = (b >= 2) ? k_proj[(size_t)(b - 2) * KV_DIM + c] : ((p >= 2) ? k_ring[((p - 2) & 15) * KV_DIM + c] : 0.0f);
        float h3 = (b >= 3) ? k_proj[(size_t)(b - 3) * KV_DIM + c] : ((p >= 3) ? k_ring[((p - 3) & 15) * KV_DIM + c] : 0.0f);
        const float* dt = k_taps + (size_t)b * 32 + h * 4;
        float acc = cur * (__bfloat162float(k_sw[0 * KV_DIM + c]) + dt[0])
                  + h1  * (__bfloat162float(k_sw[1 * KV_DIM + c]) + dt[1])
                  + h2  * (__bfloat162float(k_sw[2 * KV_DIM + c]) + dt[2])
                  + h3  * (__bfloat162float(k_sw[3 * KV_DIM + c]) + dt[3]);
        float val = (acc + cur) * INV_SQRT_9;
        float sq = warp_sum(val * val);
        if (lane == 0) sm[warp] = sq;
        __syncthreads();
        if (tid == 0) factor = fminf(QK_CLIP / (sqrtf(sm[0] + sm[1] + sm[2] + sm[3]) + 1e-6f), 1.0f);
        __syncthreads();
        store_kv_head_128<KV_FMT>(val * factor, p % kv_cap, h, tid, k_cache, k_scale, sm);
    } else {
        int h = kind_id - 56;
        int c = h * HEAD_DIM + tid;
        float cur = v_proj[(size_t)b * KV_DIM + c];
        float h1 = (b >= 1) ? v_proj[(size_t)(b - 1) * KV_DIM + c] : ((p >= 1) ? v_ring[((p - 1) & 15) * KV_DIM + c] : 0.0f);
        float h2 = (b >= 2) ? v_proj[(size_t)(b - 2) * KV_DIM + c] : ((p >= 2) ? v_ring[((p - 2) & 15) * KV_DIM + c] : 0.0f);
        float h3 = (b >= 3) ? v_proj[(size_t)(b - 3) * KV_DIM + c] : ((p >= 3) ? v_ring[((p - 3) & 15) * KV_DIM + c] : 0.0f);
        const float* dt = v_taps + (size_t)b * 32 + h * 4;
        float acc = cur * (__bfloat162float(v_sw[0 * KV_DIM + c]) + dt[0])
                  + h1  * (__bfloat162float(v_sw[1 * KV_DIM + c]) + dt[1])
                  + h2  * (__bfloat162float(v_sw[2 * KV_DIM + c]) + dt[2])
                  + h3  * (__bfloat162float(v_sw[3 * KV_DIM + c]) + dt[3]);
        float val = (acc + cur) * INV_SQRT_9;
        float sq = warp_sum(val * val);
        if (lane == 0) sm[warp] = sq;
        __syncthreads();
        if (tid == 0) factor = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / float(HEAD_DIM) + RMS_EPS);
        __syncthreads();
        store_kv_head_128<KV_FMT>(val * factor, p % kv_cap, h, tid, v_cache, v_scale, sm);
    }
}

// BLACKWELL TENSOR CORE FLASHATTENTION PREFILL (Global Q4/FP8/BF16 & Local SWA BF16!)
// Each CTA owns 1 KV head x Q_TOK tokens (GQA_RATIO * Q_TOK queries = NUM_WARPS x 16 exact!).
// Every KV tile (32 tokens) is loaded ONCE into SMEM and shared across 48 (local) or 96 (global) queries!
// Bulk global tiles (dist >= 1024) skip rel-bias + causal masking completely (pure TC FlashAttention!).
// Zero SMEM output reduction: each warp owns 16 queries end-to-end and writes directly to out_gated_h!
template <int KV_FMT, int Q_TOK>
__global__ void __launch_bounds__((Q_TOK * GQA_RATIO / 16) * 32)
p_flash_attn_prefill_tc_kernel(
    const float* __restrict__ qrgs,
    const __nv_bfloat16* __restrict__ rel_proj,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale,
    __half* __restrict__ out_gated_h,
    int kv_cap,
    int window,
    int B,
    const int32_t* __restrict__ d_pos) {
    constexpr int Q_MAX = Q_TOK * GQA_RATIO;     // 96 (global) or 48 (local)
    constexpr int NUM_WARPS = Q_MAX / 16;        // 6 or 3
    constexpr int THREADS = NUM_WARPS * 32;      // 192 or 96
    constexpr int QRGS_STRIDE = Q_DIM + R_DIM + Q_DIM + NUM_HEADS;

    int kvh = blockIdx.x;
    int b_start = blockIdx.y * Q_TOK;
    int q_tok_cnt = min(Q_TOK, B - b_start);
    if (q_tok_cnt <= 0) return;
    int q_cnt = q_tok_cnt * GQA_RATIO;

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int grp = lane >> 2;
    int tid4 = lane & 3;

    int pos0 = *d_pos;
    int q_pos_min = pos0 + b_start;
    int q_pos_max = q_pos_min + q_tok_cnt - 1;

    // 22.0 KB SMEM (Q_TOK=32) / 20.5 KB SMEM (Q_TOK=16) per CTA.
    __shared__ __half s_k[32][HEAD_DIM + 8]; // 8,704 B
    __shared__ __half s_v[32][HEAD_DIM + 8]; // 8,704 B
    __shared__ __half s_r[Q_MAX][D_REL];     // 3,072 B (Q_TOK=32) / 1,536 B (Q_TOK=16)
    __shared__ __half s_rp[64][D_REL];       // 2,048 B

    // 1. Load R[0..q_cnt-1][0..15] into SMEM once for all KV tiles
    for (int i = tid; i < q_cnt * D_REL; i += THREADS) {
        int qi = i >> 4;
        int k = i & 15;
        int h = kvh * GQA_RATIO + (qi % GQA_RATIO);
        int b = b_start + (qi / GQA_RATIO);
        s_r[qi][k] = __float2half(qrgs[(size_t)b * QRGS_STRIDE + Q_DIM + h * D_REL + k]);
    }

    // 2. Load this warp's 16 queries (q0, q1) into registers + initialize softmax with attention sink!
    int q0 = warp * 16 + grp;
    int q1 = warp * 16 + grp + 8;
    const float* qp0 = nullptr;
    const float* qp1 = nullptr;
    const float* gp0 = nullptr;
    const float* gp1 = nullptr;
    int b0 = 0, h0 = 0, pos_q0 = q_pos_min;
    int b1 = 0, h1 = 0, pos_q1 = q_pos_min;
    float m0 = 0.0f, m1 = 0.0f;
    float d0 = 1.0f, d1 = 1.0f;

    if (q0 < q_cnt) {
        h0 = kvh * GQA_RATIO + (q0 % GQA_RATIO);
        b0 = b_start + (q0 / GQA_RATIO);
        pos_q0 = pos0 + b0;
        const float* base0 = qrgs + (size_t)b0 * QRGS_STRIDE;
        qp0 = base0 + h0 * HEAD_DIM;
        gp0 = base0 + Q_DIM + R_DIM + h0 * HEAD_DIM;
        m0 = base0[Q_DIM + R_DIM + Q_DIM + h0];
    }
    if (q1 < q_cnt) {
        h1 = kvh * GQA_RATIO + (q1 % GQA_RATIO);
        b1 = b_start + (q1 / GQA_RATIO);
        pos_q1 = pos0 + b1;
        const float* base1 = qrgs + (size_t)b1 * QRGS_STRIDE;
        qp1 = base1 + h1 * HEAD_DIM;
        gp1 = base1 + Q_DIM + R_DIM + h1 * HEAD_DIM;
        m1 = base1[Q_DIM + R_DIM + Q_DIM + h1];
    }

    uint32_t q_reg[8][4];
    #pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
        int k_off = kt * 16;
        if (qp0) {
            __half2 h_a = __floats2half2_rn(qp0[k_off + tid4 * 2] * INV_SQRT_HEAD_DIM,
                                            qp0[k_off + tid4 * 2 + 1] * INV_SQRT_HEAD_DIM);
            __half2 h_c = __floats2half2_rn(qp0[k_off + tid4 * 2 + 8] * INV_SQRT_HEAD_DIM,
                                            qp0[k_off + tid4 * 2 + 9] * INV_SQRT_HEAD_DIM);
            q_reg[kt][0] = *reinterpret_cast<const uint32_t*>(&h_a);
            q_reg[kt][2] = *reinterpret_cast<const uint32_t*>(&h_c);
        } else {
            q_reg[kt][0] = 0u;
            q_reg[kt][2] = 0u;
        }
        if (qp1) {
            __half2 h_b = __floats2half2_rn(qp1[k_off + tid4 * 2] * INV_SQRT_HEAD_DIM,
                                            qp1[k_off + tid4 * 2 + 1] * INV_SQRT_HEAD_DIM);
            __half2 h_d = __floats2half2_rn(qp1[k_off + tid4 * 2 + 8] * INV_SQRT_HEAD_DIM,
                                            qp1[k_off + tid4 * 2 + 9] * INV_SQRT_HEAD_DIM);
            q_reg[kt][1] = *reinterpret_cast<const uint32_t*>(&h_b);
            q_reg[kt][3] = *reinterpret_cast<const uint32_t*>(&h_d);
        } else {
            q_reg[kt][1] = 0u;
            q_reg[kt][3] = 0u;
        }
    }

    float o[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.0f;
    }

    int c0 = tid4 * 2;
    int c1 = tid4 * 2 + 1;
    int c4 = tid4 * 2 + 8;
    int c5 = tid4 * 2 + 9;
    int k_ld_col = (lane & 8);

    int t_start = (window > 0) ? max(0, q_pos_min + 1 - window) : 0;
    int t_end = q_pos_max + 1;

    for (int t0 = t_start; t0 < t_end; t0 += 32) {
        int tile_len = min(32, t_end - t0);
        bool is_bulk = (window == 0) && (tile_len == 32) && (t0 + 31 <= q_pos_min - REL_EXTENT);

        // Cooperatively load K[32, 128] and V[32, 128] into SMEM ONCE for all 48/96 queries!
        #pragma unroll
        for (int idx = tid; idx < 512; idx += THREADS) {
            int t_loc = idx >> 4;
            int sub = idx & 15;
            if (t_loc < tile_len) {
                int slot = (t0 + t_loc) % kv_cap;
                *reinterpret_cast<uint4*>(&s_k[t_loc][sub * 8]) =
                    load_kv_8elems_h2x4<KV_FMT>(k_cache, k_scale, slot, kvh, sub);
                *reinterpret_cast<uint4*>(&s_v[t_loc][sub * 8]) =
                    load_kv_8elems_h2x4<KV_FMT>(v_cache, v_scale, slot, kvh, sub);
            } else {
                *reinterpret_cast<uint4*>(&s_k[t_loc][sub * 8]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&s_v[t_loc][sub * 8]) = make_uint4(0, 0, 0, 0);
            }
        }

        int d_min = 0;
        if (!is_bulk) {
            d_min = max(0, q_pos_min - (t0 + tile_len - 1));
            int d_max = min(REL_EXTENT - 1, q_pos_max - t0);
            int d_span = (d_max >= d_min) ? (d_max - d_min + 1) : 0;
            for (int i = tid; i < d_span * D_REL; i += THREADS) {
                int di = i >> 4;
                int k = i & 15;
                int dist = d_min + di;
                s_rp[di][k] = __float2half(__bfloat162float(__ldg(&rel_proj[k * REL_EXTENT + dist])));
            }
        }
        __syncthreads();

        // Process the 32 KV tokens in 2 Tensor Core sub-tiles of 16 keys each
        #pragma unroll
        for (int kv_sub = 0; kv_sub < 2; ++kv_sub) {
            int t_base = kv_sub * 16;
            int w_len = min(16, max(0, tile_len - t_base));
            if (w_len <= 0) break;

            int k_ld_row = t_base + ((lane & 16) >> 1) + (lane & 7);
            int v_ld_row = t_base + (lane & 15);

            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                uint32_t kb0, kb1, kb2, kb3;
                ldmatrix_x4(kb0, kb1, kb2, kb3, &s_k[k_ld_row][kt * 16 + k_ld_col]);
                mma_m16n8k16_f32_f16(s0, s1, s2, s3,
                                     q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
                                     kb0, kb1, s0, s1, s2, s3);
                mma_m16n8k16_f32_f16(s4, s5, s6, s7,
                                     q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
                                     kb2, kb3, s4, s5, s6, s7);
            }

            if (!is_bulk) {
                auto apply_rel = [&](float& s, int col, int qi, int pos_q) {
                    if (qi >= q_cnt || col >= w_len) { s = -1e30f; return; }
                    int t = t0 + t_base + col;
                    if (t > pos_q || (window > 0 && t < pos_q + 1 - window)) { s = -1e30f; return; }
                    int dist = pos_q - t;
                    if (dist < REL_EXTENT) {
                        const __half2* rh = reinterpret_cast<const __half2*>(&s_r[qi][0]);
                        const __half2* ph = reinterpret_cast<const __half2*>(&s_rp[dist - d_min][0]);
                        __half2 a0 = __hmul2(rh[0], ph[0]), a1 = __hmul2(rh[1], ph[1]);
                        a0 = __hfma2(rh[2], ph[2], a0); a1 = __hfma2(rh[3], ph[3], a1);
                        a0 = __hfma2(rh[4], ph[4], a0); a1 = __hfma2(rh[5], ph[5], a1);
                        a0 = __hfma2(rh[6], ph[6], a0); a1 = __hfma2(rh[7], ph[7], a1);
                        float2 f = __half22float2(__hadd2(a0, a1));
                        s += (f.x + f.y);
                    }
                };
                apply_rel(s0, c0, q0, pos_q0);
                apply_rel(s1, c1, q0, pos_q0);
                apply_rel(s4, c4, q0, pos_q0);
                apply_rel(s5, c5, q0, pos_q0);
                apply_rel(s2, c0, q1, pos_q1);
                apply_rel(s3, c1, q1, pos_q1);
                apply_rel(s6, c4, q1, pos_q1);
                apply_rel(s7, c5, q1, pos_q1);
            }

            float rmax0 = fmaxf(fmaxf(s0, s1), fmaxf(s4, s5));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 2));
            float rmax1 = fmaxf(fmaxf(s2, s3), fmaxf(s6, s7));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 2));

            float m0_new = fmaxf(m0, rmax0);
            float m1_new = fmaxf(m1, rmax1);
            float alpha0 = __expf(m0 - m0_new);
            float alpha1 = __expf(m1 - m1_new);

            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                o[nt][0] *= alpha0;
                o[nt][1] *= alpha0;
                o[nt][2] *= alpha1;
                o[nt][3] *= alpha1;
            }

            float ps0 = (is_bulk || s0 > -1e29f) ? __expf(s0 - m0_new) : 0.0f;
            float ps1 = (is_bulk || s1 > -1e29f) ? __expf(s1 - m0_new) : 0.0f;
            float ps4 = (is_bulk || s4 > -1e29f) ? __expf(s4 - m0_new) : 0.0f;
            float ps5 = (is_bulk || s5 > -1e29f) ? __expf(s5 - m0_new) : 0.0f;
            float ps2 = (is_bulk || s2 > -1e29f) ? __expf(s2 - m1_new) : 0.0f;
            float ps3 = (is_bulk || s3 > -1e29f) ? __expf(s3 - m1_new) : 0.0f;
            float ps6 = (is_bulk || s6 > -1e29f) ? __expf(s6 - m1_new) : 0.0f;
            float ps7 = (is_bulk || s7 > -1e29f) ? __expf(s7 - m1_new) : 0.0f;

            float rsum0 = ps0 + ps1 + ps4 + ps5;
            rsum0 += __shfl_xor_sync(0xffffffffu, rsum0, 1);
            rsum0 += __shfl_xor_sync(0xffffffffu, rsum0, 2);
            float rsum1 = ps2 + ps3 + ps6 + ps7;
            rsum1 += __shfl_xor_sync(0xffffffffu, rsum1, 1);
            rsum1 += __shfl_xor_sync(0xffffffffu, rsum1, 2);
            d0 = d0 * alpha0 + rsum0; m0 = m0_new;
            d1 = d1 * alpha1 + rsum1; m1 = m1_new;

            __half2 hp0 = __floats2half2_rn(ps0, ps1);
            __half2 hp1 = __floats2half2_rn(ps2, ps3);
            __half2 hp2 = __floats2half2_rn(ps4, ps5);
            __half2 hp3 = __floats2half2_rn(ps6, ps7);
            uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&hp0);
            uint32_t pa1 = *reinterpret_cast<const uint32_t*>(&hp1);
            uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&hp2);
            uint32_t pa3 = *reinterpret_cast<const uint32_t*>(&hp3);

            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                uint32_t vb0, vb1;
                ldmatrix_x2_trans(vb0, vb1, &s_v[v_ld_row][nt * 8]);
                mma_m16n8k16_f32_f16(o[nt][0], o[nt][1], o[nt][2], o[nt][3],
                                     pa0, pa1, pa2, pa3,
                                     vb0, vb1,
                                     o[nt][0], o[nt][1], o[nt][2], o[nt][3]);
            }
        }
        __syncthreads();
    }

    // 3. Direct gated FP16 writeback (zero SMEM reduction!)
    if (q0 < q_cnt) {
        float inv_d0 = 1.0f / d0;
        __half* dst0 = out_gated_h + (size_t)b0 * Q_DIM + h0 * HEAD_DIM;
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            int col0 = nt * 8 + c0;
            int col1 = nt * 8 + c1;
            dst0[col0] = __float2half((o[nt][0] * inv_d0) * fast_sigmoid(gp0[col0]));
            dst0[col1] = __float2half((o[nt][1] * inv_d0) * fast_sigmoid(gp0[col1]));
        }
    }
    if (q1 < q_cnt) {
        float inv_d1 = 1.0f / d1;
        __half* dst1 = out_gated_h + (size_t)b1 * Q_DIM + h1 * HEAD_DIM;
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            int col0 = nt * 8 + c0;
            int col1 = nt * 8 + c1;
            dst1[col0] = __float2half((o[nt][2] * inv_d1) * fast_sigmoid(gp1[col0]));
            dst1[col1] = __float2half((o[nt][3] * inv_d1) * fast_sigmoid(gp1[col1]));
        }
    }
}

} // namespace

void launch_prefill_warmup_cublas(cudaStream_t stream) {
    cublasHandle_t h = get_cublas(stream);
    void* d_tmp = nullptr;
    if (cudaMalloc(&d_tmp, 2048) == cudaSuccess && d_tmp) {
        cudaMemsetAsync(d_tmp, 0, 2048, stream);
        float alpha = 1.0f, beta = 0.0f;
        const __half* a = reinterpret_cast<const __half*>(d_tmp);
        const __half* b = a + 256;
        float* c = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(d_tmp) + 1024);
        cublasGemmEx(
            h, CUBLAS_OP_T, CUBLAS_OP_N,
            16, 16, 16,
            &alpha,
            a, CUDA_R_16F, 16,
            b, CUDA_R_16F, 16,
            &beta,
            c, CUDA_R_32F, 16,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaStreamSynchronize(stream);
        cudaFree(d_tmp);
    }
}

void launch_prefill_f32_to_f16(const float* src, __half* dst, int count, cudaStream_t stream) {
    int n4 = (count + 3) / 4;
    p_f32_to_f16_kernel<<<(n4 + 255) / 256, 256, 0, stream>>>(src, dst, (size_t)count);
}

void launch_prefill_gemm_h(
    const LinearWeight& w,
    const __half* x_h,
    float* y,
    __half* w_f16_scratch,
    int B,
    cudaStream_t stream) {
    if (w.kind == WeightKind::NVFP4) {
        size_t total_blocks32 = (size_t)w.rows * (w.cols >> 5);
        p_dequant_nvfp4_to_f16_kernel<<<(int)((total_blocks32 + 255) / 256), 256, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.scales, w_f16_scratch, total_blocks32);
    } else if (w.kind == WeightKind::FP8) {
        size_t total_pairs = (size_t)w.rows * (w.cols >> 1);
        p_dequant_fp8_to_f16_kernel<<<(int)((total_pairs + 255) / 256), 256, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.row_scales, w_f16_scratch, w.rows, w.cols);
    } else {
        size_t n = (size_t)w.rows * w.cols;
        p_dequant_bf16_to_f16_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(w.data), w_f16_scratch, n);
    }

    float alpha = (w.kind == WeightKind::NVFP4) ? w.global_scale : 1.0f, beta = 0.0f;
    cublasHandle_t h = get_cublas(stream);
    cublasStatus_t st = cublasGemmEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N,
        w.rows, B, w.cols,
        &alpha,
        w_f16_scratch, CUDA_R_16F, w.cols,
        x_h, CUDA_R_16F, w.cols,
        &beta,
        y, CUDA_R_32F, w.rows,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasGemmEx failed: %d (N=%d B=%d K=%d)\n", (int)st, w.rows, B, w.cols);
        std::exit(1);
    }
}

void launch_prefill_sconv_post_res(
    const float* src_or_slots, int is_slotted, const __nv_bfloat16* sconv_w, float* ring,
    const float* residual, float* out_res, float* pre_tap_out,
    float* blocks, const float* partial, int block_idx, int is_boundary,
    int B, const int32_t* d_pos,
    const float* abl_r, const float* abl_d, const float* abl_p, cudaStream_t stream) {
    int write_conv_only = (abl_r != nullptr) ? 1 : 0;
    int ty = (B + SCONV_T_BLK - 1) / SCONV_T_BLK;
    int update_ring = (ty == 1) ? 1 : 0;
    int gx = ((HIDDEN >> 2) + 255) / 256;
    p_sconv_post_res_kernel<<<dim3(gx, ty), 256, 0, stream>>>(
        src_or_slots, is_slotted, sconv_w, ring, residual, out_res, pre_tap_out,
        blocks, partial, block_idx, is_boundary, write_conv_only, update_ring, B, d_pos);
    if (!update_ring) {
        p_sconv_ring_commit_kernel<<<gx, 256, 0, stream>>>(
            src_or_slots, is_slotted, ring, B, d_pos);
    }
    if (abl_r != nullptr) {
        p_ablate_post_res_kernel<<<B, 576, 0, stream>>>(
            out_res, residual, pre_tap_out, blocks, partial, block_idx, is_boundary,
            abl_r, abl_d, abl_p);
    }
}

void launch_prefill_copy_taps(const float* src, float* dst_taps, int tap_slot, int B, cudaStream_t stream) {
    int total4 = (B * HIDDEN) >> 2;
    p_copy_taps_chunked_kernel<<<(total4 + 255) / 256, 256, 0, stream>>>(src, dst_taps, tap_slot, B);
}

void launch_prefill_dspark_commit(
    const DsparkWeights& dw, const float* taps,
    void* ctx_k[MTP_LAYERS], void* ctx_v[MTP_LAYERS],
    __half* ctx_ks[MTP_LAYERS], __half* ctx_vs[MTP_LAYERS],
    int ctx_cap[MTP_LAYERS], KVFormat global_kv_fmt,
    float* f_ring, __half* scratch_mix_h, float* scratch_kv, __half* w_f16_scratch,
    int B, const int32_t* d_pos, cudaStream_t stream) {
    p_dspark_mix_h_kernel<<<dim3(B, 4), 576, 0, stream>>>(
        taps, B, dw.fuse_alpha,
        dw.layers[0].dflare_norm, dw.layers[1].dflare_norm, dw.layers[2].dflare_norm,
        dw.markov_dflare_norm, scratch_mix_h, f_ring, d_pos);
    for (int l = 0; l < MTP_LAYERS; ++l) {
        const __half* mix_lh = scratch_mix_h + (size_t)l * B * HIDDEN;
        launch_prefill_gemm_h(dw.layers[l].w_kv_ctx, mix_lh, scratch_kv, w_f16_scratch, B, stream);
        int cap = ctx_cap[l];
        int b_off = (B > cap) ? (B - cap) : 0;
        int n_store = B - b_off;
        KVFormat lfmt = dw.layers[l].is_local ? KVFormat::BF16 : global_kv_fmt;
        if (lfmt == KVFormat::FP8) {
            p_dspark_store_ctx_kv_kernel<1><<<dim3(n_store, NUM_KV_HEADS), HEAD_DIM, 0, stream>>>(
                scratch_kv, dw.layers[l].k_norm_ctx, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], cap, b_off, d_pos);
        } else if (lfmt == KVFormat::Q4) {
            p_dspark_store_ctx_kv_kernel<2><<<dim3(n_store, NUM_KV_HEADS), HEAD_DIM, 0, stream>>>(
                scratch_kv, dw.layers[l].k_norm_ctx, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], cap, b_off, d_pos);
        } else {
            p_dspark_store_ctx_kv_kernel<0><<<dim3(n_store, NUM_KV_HEADS), HEAD_DIM, 0, stream>>>(
                scratch_kv, dw.layers[l].k_norm_ctx, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], cap, b_off, d_pos);
        }
    }
}

void launch_prefill_local_sconv_prep_store(
    float* qrgs, float* k_proj, float* v_proj,
    const __nv_bfloat16* k_sconv_w, const __nv_bfloat16* v_sconv_w,
    float* k_ring, float* v_ring,
    const __nv_bfloat16* q_clip_scale, const __nv_bfloat16* rel_clip_scale,
    __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    int kv_cap, int B, const int32_t* d_pos, cudaStream_t stream) {
    p_local_sconv_prep_store_kernel<<<dim3(64, B), 128, 0, stream>>>(
        qrgs, k_proj, v_proj, k_sconv_w, v_sconv_w, k_ring, v_ring,
        q_clip_scale, rel_clip_scale, k_cache, v_cache, kv_cap, d_pos);
    p_kv_ring_commit_kernel<<<dim3((KV_DIM >> 2) / 256, 2), 256, 0, stream>>>(
        k_proj, v_proj, k_ring, v_ring, B, d_pos);
}

void launch_prefill_global_sconv_prep_store(
    float* qrgs, float* k_proj, float* v_proj,
    const __half* in_k_h, const __half* in_v_h,
    const __nv_bfloat16* k_sconv_w, const __nv_bfloat16* v_sconv_w,
    const __nv_bfloat16* k_sconv_proj, const __nv_bfloat16* v_sconv_proj,
    float* k_ring, float* v_ring,
    const __nv_bfloat16* q_clip_scale, const __nv_bfloat16* rel_clip_scale,
    void* k_cache, void* v_cache, __half* k_scale, __half* v_scale,
    KVFormat kv_fmt, int kv_cap,
    float* scratch_taps, __half* w_f16_scratch,
    int B, const int32_t* d_pos, cudaStream_t stream) {
    // 1. Batched cuBLAS FP16 TC dynamic tap projections ([32, 2304] x [B, 2304])
    float* k_taps = scratch_taps;
    float* v_taps = scratch_taps + (size_t)B * 32;
    LinearWeight kw = {WeightKind::BF16, 32, HIDDEN, const_cast<__nv_bfloat16*>(k_sconv_proj), nullptr, nullptr, 1.0f};
    LinearWeight vw = {WeightKind::BF16, 32, HIDDEN, const_cast<__nv_bfloat16*>(v_sconv_proj), nullptr, nullptr, 1.0f};
    launch_prefill_gemm_h(kw, in_k_h, k_taps, w_f16_scratch, B, stream);
    launch_prefill_gemm_h(vw, in_v_h, v_taps, w_f16_scratch, B, stream);

    // 2. 1-launch 2D-parallel fused dynamic sconv + Q/R/K clip (log-pos tau) + V RMSNorm + quantized KV store!
    dim3 grid(64, B);
    if (kv_fmt == KVFormat::FP8) {
        p_global_attn_prep_store_kernel<1><<<grid, 128, 0, stream>>>(
            qrgs, k_proj, v_proj, k_sconv_w, v_sconv_w, k_taps, v_taps, k_ring, v_ring,
            q_clip_scale, rel_clip_scale, k_cache, v_cache, k_scale, v_scale, kv_cap, d_pos);
    } else if (kv_fmt == KVFormat::Q4) {
        p_global_attn_prep_store_kernel<2><<<grid, 128, 0, stream>>>(
            qrgs, k_proj, v_proj, k_sconv_w, v_sconv_w, k_taps, v_taps, k_ring, v_ring,
            q_clip_scale, rel_clip_scale, k_cache, v_cache, k_scale, v_scale, kv_cap, d_pos);
    } else {
        p_global_attn_prep_store_kernel<0><<<grid, 128, 0, stream>>>(
            qrgs, k_proj, v_proj, k_sconv_w, v_sconv_w, k_taps, v_taps, k_ring, v_ring,
            q_clip_scale, rel_clip_scale, k_cache, v_cache, k_scale, v_scale, kv_cap, d_pos);
    }
    p_kv_ring_commit_kernel<<<dim3((KV_DIM >> 2) / 256, 2), 256, 0, stream>>>(
        k_proj, v_proj, k_ring, v_ring, B, d_pos);
}

void launch_prefill_flash_attn_tc(
    const float* qrgs, const __nv_bfloat16* rel_proj,
    const void* k_cache, const void* v_cache,
    const __half* k_scale, const __half* v_scale,
    KVFormat kv_fmt, __half* out_gated_h,
    int kv_cap, int window, int B, const int32_t* d_pos, cudaStream_t stream) {
    if (window > 0) {
        // Local SWA (BF16): Q_TOK=16 (48 queries/CTA, 3 warps=96 threads -> 128 CTAs for B=256 = 1 wave!)
        dim3 grid(NUM_KV_HEADS, (B + 15) / 16);
        p_flash_attn_prefill_tc_kernel<0, 16><<<grid, 96, 0, stream>>>(
            qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, out_gated_h, kv_cap, window, B, d_pos);
    } else {
        // Global (BF16/FP8/Q4): Q_TOK=32 (96 queries/CTA, 6 warps=192 threads -> 128 CTAs for B=512 = 1 wave!)
        dim3 grid(NUM_KV_HEADS, (B + 31) / 32);
        if (kv_fmt == KVFormat::FP8) {
            p_flash_attn_prefill_tc_kernel<1, 32><<<grid, 192, 0, stream>>>(
                qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, out_gated_h, kv_cap, 0, B, d_pos);
        } else if (kv_fmt == KVFormat::Q4) {
            p_flash_attn_prefill_tc_kernel<2, 32><<<grid, 192, 0, stream>>>(
                qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, out_gated_h, kv_cap, 0, B, d_pos);
        } else {
            p_flash_attn_prefill_tc_kernel<0, 32><<<grid, 192, 0, stream>>>(
                qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, out_gated_h, kv_cap, 0, B, d_pos);
        }
    }
}

void launch_prefill_controller_one(
    const float* x, const float* saved, const float* prior_last, const float* cp_q,
    const ControllerWeights& cw, const __nv_bfloat16* post_norm,
    float* out, __half* out_h,
    float* scratch_msg, __half* scratch_cat_h, float* scratch_d, __half* scratch_d_h,
    float* scratch_f, float* scratch_g, __half* w_f16_scratch,
    int B, cudaStream_t stream) {
    p_ctrl_prep_h_kernel<<<B, 576, 0, stream>>>(x, saved, prior_last, cp_q, scratch_msg, scratch_cat_h);

    LinearWeight down_w = cw.down_nvfp4;
    LinearWeight forget_w = cw.forget_nvfp4;
    LinearWeight gain_w = cw.gain_nvfp4;
    if (down_w.data == nullptr) {
        down_w = {WeightKind::BF16, GATE_RANK, 2 * HIDDEN, cw.down_w, nullptr, nullptr, 1.0f};
        forget_w = {WeightKind::BF16, HIDDEN, GATE_RANK, cw.forget_w, nullptr, nullptr, 1.0f};
        gain_w = {WeightKind::BF16, HIDDEN, GATE_RANK, cw.gain_w, nullptr, nullptr, 1.0f};
    }

    launch_prefill_gemm_h(down_w, scratch_cat_h, scratch_d, w_f16_scratch, B, stream);
    launch_prefill_f32_to_f16(scratch_d, scratch_d_h, B * GATE_RANK, stream);
    launch_prefill_gemm_h(forget_w, scratch_d_h, scratch_f, w_f16_scratch, B, stream);
    launch_prefill_gemm_h(gain_w, scratch_d_h, scratch_g, w_f16_scratch, B, stream);

    p_ctrl_finish_kernel<<<B, 576, 0, stream>>>(x, scratch_msg, scratch_f, scratch_g, post_norm, out, out_h);
}

void launch_prefill_route_and_sort(
    const float* router_logits, const float* router_bias,
    int32_t* route_ids, float* route_w,
    int32_t* expert_offsets, int32_t* sorted_tok, int32_t* sorted_slot, float* sorted_w,
    int B, cudaStream_t stream) {
    int TR = (B <= 512) ? 32 : 128;
    p_route_topk_only_kernel<<<B, 32, 0, stream>>>(router_logits, router_bias, route_ids, route_w);
    p_sort_experts_kernel<<<1, NUM_EXPERTS, 0, stream>>>(
        route_ids, route_w, B, TR, expert_offsets, sorted_tok, sorted_slot, sorted_w);
}

void launch_prefill_moe_gate_up_h(
    const ExpertBank& w13, const __half* x_h,
    const int32_t* expert_offsets, const int32_t* sorted_tok, const int32_t* sorted_slot,
    __half* out_act_h, int B, cudaStream_t stream) {
    if (w13.is_nvfp4) {
        if (B <= 512) dispatch_wgemm_tm<0, WgemmMode::Up, HIDDEN, 32>(w13, x_h, expert_offsets, sorted_tok, sorted_slot, nullptr, out_act_h, nullptr, stream);
        else          dispatch_wgemm_tm<0, WgemmMode::Up, HIDDEN, 128>(w13, x_h, expert_offsets, sorted_tok, sorted_slot, nullptr, out_act_h, nullptr, stream);
    } else {
        if (B <= 512) dispatch_wgemm_tm<1, WgemmMode::Up, HIDDEN, 32>(w13, x_h, expert_offsets, sorted_tok, sorted_slot, nullptr, out_act_h, nullptr, stream);
        else          dispatch_wgemm_tm<1, WgemmMode::Up, HIDDEN, 128>(w13, x_h, expert_offsets, sorted_tok, sorted_slot, nullptr, out_act_h, nullptr, stream);
    }
}

void launch_prefill_moe_down_h(
    const ExpertBank& w2, const __half* act_h,
    const int32_t* expert_offsets, const int32_t* sorted_tok, const int32_t* sorted_slot,
    const float* sorted_w, float* out_down, int B, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(out_down, 0, (size_t)B * HIDDEN * sizeof(float), stream));
    if (w2.is_nvfp4) {
        if (B <= 512) dispatch_wgemm_tm<0, WgemmMode::Down, INTERMEDIATE, 32>(w2, act_h, expert_offsets, sorted_tok, sorted_slot, sorted_w, nullptr, out_down, stream);
        else          dispatch_wgemm_tm<0, WgemmMode::Down, INTERMEDIATE, 128>(w2, act_h, expert_offsets, sorted_tok, sorted_slot, sorted_w, nullptr, out_down, stream);
    } else {
        if (B <= 512) dispatch_wgemm_tm<1, WgemmMode::Down, INTERMEDIATE, 32>(w2, act_h, expert_offsets, sorted_tok, sorted_slot, sorted_w, nullptr, out_down, stream);
        else          dispatch_wgemm_tm<1, WgemmMode::Down, INTERMEDIATE, 128>(w2, act_h, expert_offsets, sorted_tok, sorted_slot, sorted_w, nullptr, out_down, stream);
    }
}

} // namespace squid
