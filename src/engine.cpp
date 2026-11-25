#include "engine.hpp"

#include "tiger.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace squid {

namespace {

struct PrefillPoolLayout {
    uint64_t off_moe_slot_out;
    uint64_t off_moe_act_h;
    uint64_t off_mlp_normed_h;
    uint64_t off_router_logits;
    uint64_t off_w_f16;
    uint64_t off_ctrl_w_f16;
    uint64_t off_attn_in_h[3];
    uint64_t off_attn_in[3];
    uint64_t off_mlp_in;
    uint64_t off_ctrl_msg;
    uint64_t off_ctrl_cat_h;
    uint64_t off_ctrl_d;
    uint64_t off_ctrl_d_h;
    uint64_t off_ctrl_f;
    uint64_t off_ctrl_g;
    uint64_t off_qrgs;
    uint64_t off_k_proj;
    uint64_t off_v_proj;
    uint64_t off_attn_gated_h;
    uint64_t off_o_proj;
    uint64_t off_mlp_normed;
    uint64_t total_bytes;
};

uint64_t compute_dspark_ws_floats(const ModelWeights& mw) {
    if (!mw.dspark.loaded) return 0;
    const uint64_t vd = static_cast<uint64_t>(mw.dspark.draft_vocab_size);
    constexpr uint64_t kFixedFloats = 393216;
    const uint64_t base_logits_floats = (mw.dspark.unembed_draft.kind == WeightKind::NVFP4)
                                            ? ((static_cast<uint64_t>(MTP_MAX_DRAFTS) * vd + 1) / 2)
                                            : (static_cast<uint64_t>(MTP_MAX_DRAFTS) * vd);
    return kFixedFloats + base_logits_floats;
}

PrefillPoolLayout compute_prefill_pool_layout() {
    constexpr uint64_t a = VRAM_ALIGNMENT;
    constexpr uint64_t pb = PREFILL_BATCH_MAX;
    const uint64_t h_f16 = align_up_u64(pb * HIDDEN * 2, a);
    const uint64_t h_f32 = align_up_u64(pb * HIDDEN * 4, a);

    // Overlay A (Phase 6 MoE + Phase 0 prev_mlp_pending): 18 MiB L2-resident moe_down + 36 MiB moe_act_h!
    const uint64_t off_moe_slot_out = 0;
    const uint64_t off_moe_act_h = off_moe_slot_out + h_f32;
    const uint64_t moe_end = off_moe_act_h + align_up_u64(pb * TOP_K * INTERMEDIATE * 2, a);

    // Overlay B1 (Phase 1: pre-attn depth-norm + 3-stream controller, aliasing mlp_in & ctrl_f/g!)
    uint64_t cur = 0;
    uint64_t off_attn_in_h[3] = {cur, cur + h_f16, cur + 2 * h_f16};
    cur += 3 * h_f16;
    const uint64_t attn_in_h_end = cur;
    uint64_t off_attn_in[3] = {cur, cur + h_f32, cur + 2 * h_f32};
    cur += 3 * h_f32;
    const uint64_t off_mlp_in = off_attn_in[0]; // local MLP/attn never coexists with global 3-stream attn_in
    const uint64_t off_ctrl_msg = cur;
    cur += h_f32;
    const uint64_t off_ctrl_cat_h = cur;
    const uint64_t ctrl_cat_bytes = align_up_u64(pb * 2 * HIDDEN * 2, a);
    // ctrl_f (18 MiB) aliases ctrl_cat_h (18 MiB) after down_gemm reads ctrl_cat_h; ctrl_g (18 MiB) follows.
    const uint64_t off_ctrl_f = off_ctrl_cat_h;
    const uint64_t off_ctrl_g = off_ctrl_f + h_f32;
    cur += std::max(ctrl_cat_bytes, 2 * h_f32);
    const uint64_t off_ctrl_d = cur;
    cur += align_up_u64(pb * GATE_RANK * 4, a);
    const uint64_t off_ctrl_d_h = cur;
    cur += align_up_u64(pb * GATE_RANK * 2, a);
    const uint64_t off_ctrl_w_f16 = cur;
    const uint64_t ctrl_w_bytes = align_up_u64(static_cast<uint64_t>(GATE_RANK) * 2 * HIDDEN * 2, a);
    cur += ctrl_w_bytes;
    const uint64_t b1_end = cur;

    // Overlay B2 (Phase 2 & 3: QKV GEMMs, sconv, FlashAttention + dense weight scratch)
    uint64_t cur2 = attn_in_h_end;
    const uint64_t off_qrgs = cur2;
    cur2 += align_up_u64(pb * QRGS_OUT_DIM * 4, a);
    const uint64_t off_k_proj = cur2;
    cur2 += align_up_u64(pb * KV_DIM * 4, a);
    const uint64_t off_v_proj = cur2;
    cur2 += align_up_u64(pb * KV_DIM * 4, a);
    const uint64_t off_attn_gated_h = cur2;
    cur2 += align_up_u64(pb * Q_DIM * 2, a);
    const uint64_t off_w_f16 = cur2;
    const uint64_t w_f16_bytes = align_up_u64(static_cast<uint64_t>(QRGS_OUT_DIM) * HIDDEN * 2, a);
    const uint64_t b2_end = off_w_f16 + w_f16_bytes;
    SQUID_ASSERT(b2_end <= off_ctrl_d); // keeps p_ctrl_d free as scratch_taps during global sconv!

    // Overlay B3 (Phase 4 & 5: O proj, attn post, final_norm)
    const uint64_t off_o_proj = 0;
    const uint64_t off_mlp_normed = 0; // only used at visit 47 final_norm after all visits finish
    SQUID_ASSERT(off_o_proj + h_f32 <= off_attn_gated_h);

    const uint64_t off_mlp_normed_h = std::max({moe_end, b1_end, b2_end});
    const uint64_t off_router_logits = off_mlp_normed_h + h_f16;
    const uint64_t total_bytes = off_router_logits + align_up_u64(pb * NUM_EXPERTS * 4, a);

    PrefillPoolLayout l{};
    l.off_moe_slot_out = off_moe_slot_out;
    l.off_moe_act_h = off_moe_act_h;
    l.off_mlp_normed_h = off_mlp_normed_h;
    l.off_router_logits = off_router_logits;
    l.off_w_f16 = off_w_f16;
    l.off_ctrl_w_f16 = off_ctrl_w_f16;
    for (int i = 0; i < 3; ++i) {
        l.off_attn_in_h[i] = off_attn_in_h[i];
        l.off_attn_in[i] = off_attn_in[i];
    }
    l.off_mlp_in = off_mlp_in;
    l.off_ctrl_msg = off_ctrl_msg;
    l.off_ctrl_cat_h = off_ctrl_cat_h;
    l.off_ctrl_d = off_ctrl_d;
    l.off_ctrl_d_h = off_ctrl_d_h;
    l.off_ctrl_f = off_ctrl_f;
    l.off_ctrl_g = off_ctrl_g;
    l.off_qrgs = off_qrgs;
    l.off_k_proj = off_k_proj;
    l.off_v_proj = off_v_proj;
    l.off_attn_gated_h = off_attn_gated_h;
    l.off_o_proj = off_o_proj;
    l.off_mlp_normed = off_mlp_normed;
    l.total_bytes = total_bytes;
    return l;
}

uint64_t compute_runtime_vram_bytes(
    const VisitInfo schedule[NUM_VISITS],
    uint32_t context_tokens_max,
    KVFormat kv_format,
    bool has_dspark,
    uint64_t draft_vocab_count,
    uint64_t dspark_ws_floats) {
    constexpr uint64_t a = VRAM_ALIGNMENT;
    uint64_t total = 0;

    for (int vi = 0; vi < NUM_VISITS; ++vi) {
        const bool is_glob = schedule[vi].is_global;
        const uint32_t cap = is_glob ? context_tokens_max : (SWA_WINDOW + PREFILL_LOCAL_ATTN_SUB);
        const KVFormat vfmt = is_glob ? kv_format : KVFormat::BF16;
        total += kv_cache_pair_vram_bytes(cap, vfmt);
        total += 2 * align_up_u64(static_cast<uint64_t>(SCONV_RING_SLOTS) * KV_DIM * 4, a);
        total += 2 * align_up_u64(static_cast<uint64_t>(SCONV_RING_SLOTS) * HIDDEN * 4, a);
    }

    if (has_dspark) {
        for (int ml = 0; ml < MTP_LAYERS; ++ml) {
            const bool is_loc = is_mtp_local_layer(ml);
            const uint32_t cap = is_loc ? (SWA_WINDOW + MTP_BLOCK_SIZE)
                                        : (context_tokens_max > 524288 ? 65536u : context_tokens_max);
            const KVFormat lfmt = is_loc ? KVFormat::BF16 : kv_format;
            total += kv_cache_pair_vram_bytes(cap, lfmt);
        }
        total += align_up_u64(static_cast<uint64_t>(SCONV_RING_SLOTS) * HIDDEN * 4, a);
    }

    total += align_up_u64(SPLIT_BUFFER_FLOATS * 4, a);

    constexpr uint64_t b = MTP_BLOCK_SIZE;
    total += align_up_u64(4, a); // d_pos
    total += align_up_u64(b * 4, a); // d_tokens
    total += align_up_u64(b * 4, a); // d_next_tokens
    total += align_up_u64(4, a); // d_anchor
    total += align_up_u64(sizeof(LookupDraft), a);
    if (has_dspark) {
        total += align_up_u64(static_cast<uint64_t>(MTP_MAX_DRAFTS) * 4, a);
    }

    total += 3 * align_up_u64(b * HIDDEN * 4, a);
    total += align_up_u64(b * 4 * HIDDEN * 4, a);
    total += 2 * align_up_u64(b * HIDDEN * 4, a);
    total += 3 * (align_up_u64(b * HIDDEN * 4, a) + align_up_u64(b * HIDDEN * 2, a));

    total += align_up_u64(b * QRGS_OUT_DIM * 4, a);
    total += 2 * align_up_u64(b * KV_DIM * 4, a);
    total += align_up_u64(b * Q_DIM * 4, a);
    total += align_up_u64(b * Q_DIM * 2, a);
    total += 4 * align_up_u64(b * HIDDEN * 4, a);
    total += align_up_u64(b * HIDDEN * 2, a);
    total += align_up_u64(3 * b * HIDDEN * 4, a);
    total += align_up_u64(3 * b * 2 * HIDDEN * 4, a);
    total += align_up_u64(3 * b * GATE_RANK * 4, a);
    total += align_up_u64(b * NUM_EXPERTS * 4, a);
    total += 2 * align_up_u64(b * TOP_K * 4, a);
    total += align_up_u64(b * TOP_K * INTERMEDIATE * 2, a);
    total += align_up_u64(b * HIDDEN * 4, a);
    if (has_dspark) {
        total += align_up_u64(4 * b * HIDDEN * 4, a);
        total += align_up_u64(3 * b * HIDDEN * 4, a);
        total += align_up_u64(b * 2 * KV_DIM * 4, a);
        total += align_up_u64(dspark_ws_floats * 4, a);
    }
    total += 2 * align_up_u64(b * 1024 * 4, a);
    total += align_up_u64(b * VOCAB_UNPADDED * 4, a);
    if (has_dspark) {
        total += align_up_u64(static_cast<uint64_t>(MTP_MAX_DRAFTS) * draft_vocab_count * 4, a);
        total += align_up_u64(static_cast<uint64_t>(MTP_MAX_DRAFTS) * 4, a);
    }
    total += align_up_u64(sizeof(SampleConfig), a);

    constexpr uint64_t pb = PREFILL_BATCH_MAX;
    total += align_up_u64(pb * 4, a);
    total += align_up_u64(static_cast<uint64_t>(PREFILL_SUB_CHUNKS_MAX) * 4, a);
    total += 3 * align_up_u64(pb * HIDDEN * 4, a);
    total += align_up_u64(pb * 4 * HIDDEN * 4, a);
    total += 2 * align_up_u64(pb * HIDDEN * 4, a);
    total += align_up_u64(pb * HIDDEN * 4, a);
    total += 2 * align_up_u64(pb * TOP_K * 4, a);
    total += align_up_u64(1024 * 4, a);
    total += 3 * align_up_u64(pb * TOP_K * 4, a);
    if (has_dspark) {
        total += align_up_u64(4 * pb * HIDDEN * 4, a);
    }
    total += align_up_u64(compute_prefill_pool_layout().total_bytes, a);
    return total;
}

struct GlobalKVBytes {
    size_t data;
    size_t scale;
};

GlobalKVBytes global_kv_bytes_for_tokens(KVFormat fmt, uint32_t token_count) {
    const size_t tc = token_count;
    switch (fmt) {
        case KVFormat::BF16: return {tc * KV_DIM * 2, 0};
        case KVFormat::FP8:  return {tc * KV_DIM, tc * NUM_KV_HEADS * 2};
        case KVFormat::Q4:   return {tc * (KV_DIM / 2), tc * KV_Q4_SCALES_PER_TOK * 2};
    }
    SQUID_ASSERT(false);
    return {0, 0};
}

struct AblationSlice {
    const float* r = nullptr;
    const float* d = nullptr;
    const float* p = nullptr;
};

inline AblationSlice ablation_slice(const LoadedModel& m, bool is_mlp, int vi) {
    if (!m.ablation.active) return {};
    const bool active = is_mlp ? m.ablation.mlp_nonzero[vi] : m.ablation.attn_nonzero[vi];
    if (!active) return {};
    const float* dirs = is_mlp ? m.ablation.mlp_dirs : m.ablation.attn_dirs;
    const float* det = is_mlp ? m.ablation.mlp_det : m.ablation.attn_det;
    const float* params = is_mlp ? m.ablation.mlp_params : m.ablation.attn_params;
    return {
        dirs + static_cast<size_t>(vi) * HIDDEN,
        det ? (det + static_cast<size_t>(vi) * HIDDEN) : nullptr,
        params + static_cast<size_t>(vi) * 4,
    };
}

cudaGraphExec_t end_capture_and_instantiate(cudaStream_t stream) {
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    SQUID_ASSERT(exec != nullptr);
    return exec;
}

} // namespace

bool Engine::init(const LoadedModel* loaded, uint32_t max_ctx, bool graphs, KVFormat kv_fmt) {
    SQUID_ASSERT(loaded != nullptr);
    SQUID_ASSERT(max_ctx >= 64);
    SQUID_ASSERT(arena.base_ptr == nullptr);

    model = loaded;
    context_tokens_max = max_ctx;
    use_graphs = graphs;
    kv_format = kv_fmt;
    build_visit_schedule(schedule);

    const bool has_dspark = loaded->mw.dspark.loaded;
    const uint64_t draft_vocab_count = has_dspark ? static_cast<uint64_t>(loaded->mw.dspark.draft_vocab_size) : 0;
    const uint64_t dspark_ws_floats = compute_dspark_ws_floats(loaded->mw);

    const uint64_t runtime_bytes = compute_runtime_vram_bytes(
        schedule, context_tokens_max, kv_format, has_dspark, draft_vocab_count, dspark_ws_floats);
    if (!check_spark_memory(loaded->arena.capacity_bytes, runtime_bytes)) return false;

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&aux_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&join_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming));

    arena.init(runtime_bytes);

    for (int vi = 0; vi < NUM_VISITS; ++vi) {
        const bool is_glob = schedule[vi].is_global;
        const uint32_t cap = is_glob ? context_tokens_max : (SWA_WINDOW + PREFILL_LOCAL_ATTN_SUB);
        const KVFormat vfmt = is_glob ? kv_format : KVFormat::BF16;
        kv_cap[vi] = static_cast<int>(cap);
        alloc_kv_pair(cap, vfmt, &k_cache[vi], &v_cache[vi], &k_scale[vi], &v_scale[vi]);
        k_ring[vi] = arena.alloc_many<float>(SCONV_RING_SLOTS * KV_DIM);
        v_ring[vi] = arena.alloc_many<float>(SCONV_RING_SLOTS * KV_DIM);
        attn_ring[vi] = arena.alloc_many<float>(SCONV_RING_SLOTS * HIDDEN);
        mlp_ring[vi] = arena.alloc_many<float>(SCONV_RING_SLOTS * HIDDEN);
    }

    if (has_dspark) {
        for (int ml = 0; ml < MTP_LAYERS; ++ml) {
            const bool is_loc = is_mtp_local_layer(ml);
            const uint32_t cap = is_loc ? (SWA_WINDOW + MTP_BLOCK_SIZE)
                                        : (context_tokens_max > 524288 ? 65536u : context_tokens_max);
            const KVFormat lfmt = is_loc ? KVFormat::BF16 : kv_format;
            ds_kv_cap[ml] = static_cast<int>(cap);
            alloc_kv_pair(cap, lfmt, &ds_k_cache[ml], &ds_v_cache[ml], &ds_k_scale[ml], &ds_v_scale[ml]);
        }
        ds_f_ring = arena.alloc_many<float>(SCONV_RING_SLOTS * HIDDEN);
    }

    split_buf = arena.alloc_many<float>(SPLIT_BUFFER_FLOATS);

    constexpr uint64_t b_max = MTP_BLOCK_SIZE;
    d_pos = arena.alloc_one<int32_t>();
    d_tokens = arena.alloc_many<int32_t>(b_max);
    d_next_tokens = arena.alloc_many<int32_t>(b_max);
    d_anchor = arena.alloc_one<int32_t>();
    d_lookup_draft = arena.alloc_one<LookupDraft>();
    d_drafts = d_lookup_draft->tokens;
    d_draft_is_delta = &d_lookup_draft->is_delta;
    if (has_dspark) {
        d_draft_conf = arena.alloc_many<float>(MTP_MAX_DRAFTS);
    }

    void* host_res_raw = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host_res_raw, sizeof(DecodeResult), cudaHostAllocMapped));
    h_decode_result = static_cast<DecodeResult*>(host_res_raw);
    void* dev_res_raw = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_res_raw, host_res_raw, 0));
    d_decode_result = static_cast<DecodeResult*>(dev_res_raw);

    void* h_ld_raw = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_ld_raw, sizeof(LookupDraft), cudaHostAllocDefault));
    h_lookup_draft = static_cast<LookupDraft*>(h_ld_raw);
    h_lookup_draft->is_delta = 1;

    hidden = arena.alloc_many<float>(b_max * HIDDEN);
    prelude = arena.alloc_many<float>(b_max * HIDDEN);
    partial = arena.alloc_many<float>(b_max * HIDDEN);
    blocks = arena.alloc_many<float>(b_max * 4 * HIDDEN);
    saved = arena.alloc_many<float>(b_max * HIDDEN);
    prior_last = arena.alloc_many<float>(b_max * HIDDEN);

    for (int s = 0; s < 3; ++s) {
        attn_in[s] = arena.alloc_many<float>(b_max * HIDDEN);
        attn_in_h[s] = static_cast<__half*>(arena.alloc_raw(b_max * HIDDEN * 2));
    }

    qrgs = arena.alloc_many<float>(b_max * QRGS_OUT_DIM);
    k_proj = arena.alloc_many<float>(b_max * KV_DIM);
    v_proj = arena.alloc_many<float>(b_max * KV_DIM);
    attn_gated = arena.alloc_many<float>(b_max * Q_DIM);
    attn_gated_h = static_cast<__half*>(arena.alloc_raw(b_max * Q_DIM * 2));
    o_proj = arena.alloc_many<float>(b_max * HIDDEN);
    residual = arena.alloc_many<float>(b_max * HIDDEN);
    mlp_in = arena.alloc_many<float>(b_max * HIDDEN);
    mlp_normed = arena.alloc_many<float>(b_max * HIDDEN);
    mlp_normed_h = static_cast<__half*>(arena.alloc_raw(b_max * HIDDEN * 2));
    ctrl_msg = arena.alloc_many<float>(3 * b_max * HIDDEN);
    ctrl_cat = arena.alloc_many<float>(3 * b_max * 2 * HIDDEN);
    ctrl_d = arena.alloc_many<float>(3 * b_max * GATE_RANK);
    router_logits = arena.alloc_many<float>(b_max * NUM_EXPERTS);
    route_ids = arena.alloc_many<int32_t>(b_max * TOP_K);
    route_w = arena.alloc_many<float>(b_max * TOP_K);
    moe_act_h = static_cast<__half*>(arena.alloc_raw(b_max * TOP_K * INTERMEDIATE * 2));
    moe_down = arena.alloc_many<float>(b_max * HIDDEN);
    if (has_dspark) {
        taps = arena.alloc_many<float>(4 * b_max * HIDDEN);
        ds_mix = arena.alloc_many<float>(3 * b_max * HIDDEN);
        ds_kv = arena.alloc_many<float>(b_max * 2 * KV_DIM);
        ds_ws = arena.alloc_many<float>(dspark_ws_floats);
    }
    lm_blk_val = arena.alloc_many<float>(b_max * 1024);
    lm_blk_idx = arena.alloc_many<int32_t>(b_max * 1024);
    lm_logits = arena.alloc_many<float>(b_max * VOCAB_UNPADDED);
    if (has_dspark) {
        d_draft_probs = arena.alloc_many<float>(static_cast<uint64_t>(MTP_MAX_DRAFTS) * draft_vocab_count);
        d_draft_q_sel = arena.alloc_many<float>(MTP_MAX_DRAFTS);
    }
    d_sample_cfg = arena.alloc_one<SampleConfig>();
    const SampleConfig zero_cfg{0.0f, 1.0f, 0, 0, 0};
    const int32_t zero_i32 = 0;
    CUDA_CHECK(cudaMemcpyAsync(d_sample_cfg, &zero_cfg, sizeof(SampleConfig), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_draft_is_delta, &zero_i32, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    // Batched prefill arena (B <= 2048) with phase-aliased transient pool
    constexpr uint64_t pb = PREFILL_BATCH_MAX;
    p_tokens = arena.alloc_many<int32_t>(pb);
    p_pos_vec = arena.alloc_many<int32_t>(PREFILL_SUB_CHUNKS_MAX);
    p_hidden = arena.alloc_many<float>(pb * HIDDEN);
    p_prelude = arena.alloc_many<float>(pb * HIDDEN);
    p_partial = arena.alloc_many<float>(pb * HIDDEN);
    p_blocks = arena.alloc_many<float>(pb * 4 * HIDDEN);
    p_saved = arena.alloc_many<float>(pb * HIDDEN);
    p_prior_last = arena.alloc_many<float>(pb * HIDDEN);
    p_residual = arena.alloc_many<float>(pb * HIDDEN);
    p_route_ids = arena.alloc_many<int32_t>(pb * TOP_K);
    p_route_w = arena.alloc_many<float>(pb * TOP_K);
    p_exp_offsets = arena.alloc_many<int32_t>(1024);
    p_sorted_tok = arena.alloc_many<int32_t>(pb * TOP_K);
    p_sorted_slot = arena.alloc_many<int32_t>(pb * TOP_K);
    p_sorted_w = arena.alloc_many<float>(pb * TOP_K);
    if (has_dspark) {
        p_taps = arena.alloc_many<float>(4 * pb * HIDDEN);
    }

    const PrefillPoolLayout pl = compute_prefill_pool_layout();
    auto* pool_base = static_cast<uint8_t*>(arena.alloc_raw(pl.total_bytes));
    p_moe_slot_out = reinterpret_cast<float*>(pool_base + pl.off_moe_slot_out);
    p_moe_act_h = reinterpret_cast<__half*>(pool_base + pl.off_moe_act_h);
    p_mlp_normed_h = reinterpret_cast<__half*>(pool_base + pl.off_mlp_normed_h);
    p_router_logits = reinterpret_cast<float*>(pool_base + pl.off_router_logits);
    p_w_f16 = reinterpret_cast<__half*>(pool_base + pl.off_w_f16);
    p_ctrl_w_f16 = reinterpret_cast<__half*>(pool_base + pl.off_ctrl_w_f16);
    for (int s = 0; s < 3; ++s) {
        p_attn_in_h[s] = reinterpret_cast<__half*>(pool_base + pl.off_attn_in_h[s]);
        p_attn_in[s] = reinterpret_cast<float*>(pool_base + pl.off_attn_in[s]);
    }
    p_mlp_in = reinterpret_cast<float*>(pool_base + pl.off_mlp_in);
    p_ctrl_msg = reinterpret_cast<float*>(pool_base + pl.off_ctrl_msg);
    p_ctrl_cat_h = reinterpret_cast<__half*>(pool_base + pl.off_ctrl_cat_h);
    p_ctrl_d = reinterpret_cast<float*>(pool_base + pl.off_ctrl_d);
    p_ctrl_d_h = reinterpret_cast<__half*>(pool_base + pl.off_ctrl_d_h);
    p_ctrl_f = reinterpret_cast<float*>(pool_base + pl.off_ctrl_f);
    p_ctrl_g = reinterpret_cast<float*>(pool_base + pl.off_ctrl_g);
    p_qrgs = reinterpret_cast<float*>(pool_base + pl.off_qrgs);
    p_k_proj = reinterpret_cast<float*>(pool_base + pl.off_k_proj);
    p_v_proj = reinterpret_cast<float*>(pool_base + pl.off_v_proj);
    p_attn_gated_h = reinterpret_cast<__half*>(pool_base + pl.off_attn_gated_h);
    p_o_proj = reinterpret_cast<float*>(pool_base + pl.off_o_proj);
    p_mlp_normed = reinterpret_cast<float*>(pool_base + pl.off_mlp_normed);

    arena.assert_exact_fill();
    launch_prefill_warmup_cublas(stream);

    const double rt_gib = static_cast<double>(arena.capacity_bytes) / (1024.0 * 1024.0 * 1024.0);
    printf("[engine] Initialized max_ctx=%u | kv=%s | runtime VRAM=%.2f GiB (1 arena, %u slices) | graphs=%s\n",
           context_tokens_max, kv_format_name(kv_format), rt_gib, arena.allocation_count,
           use_graphs ? "on" : "off");
    std::fflush(stdout);
    reset();
    return true;
}

void Engine::deinit() {
    if (b1_graph_exec) CUDA_CHECK(cudaGraphExecDestroy(b1_graph_exec));
    for (int i = 0; i <= MTP_MAX_DRAFTS; ++i) {
        if (spec_graph_exec[i]) CUDA_CHECK(cudaGraphExecDestroy(spec_graph_exec[i]));
        spec_graph_exec[i] = nullptr;
    }
    if (draft_graph_exec) CUDA_CHECK(cudaGraphExecDestroy(draft_graph_exec));
    if (verify_b8_graph_exec) CUDA_CHECK(cudaGraphExecDestroy(verify_b8_graph_exec));
    if (adaptive_graph) adaptive_graph_destroy(adaptive_graph);
    for (int i = 0; i < 2; ++i) {
        if (prefill_b8_graph_exec[i]) CUDA_CHECK(cudaGraphExecDestroy(prefill_b8_graph_exec[i]));
        prefill_b8_graph_exec[i] = nullptr;
    }
    if (h_decode_result) CUDA_CHECK(cudaFreeHost(h_decode_result));
    if (h_lookup_draft) CUDA_CHECK(cudaFreeHost(h_lookup_draft));
    deinit_prefix_cache();
    arena.deinit();
    if (done_event) CUDA_CHECK(cudaEventDestroy(done_event));
    if (fork_event) CUDA_CHECK(cudaEventDestroy(fork_event));
    if (join_event) CUDA_CHECK(cudaEventDestroy(join_event));
    if (aux_stream) CUDA_CHECK(cudaStreamDestroy(aux_stream));
    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
    *this = Engine{};
}

void Engine::reset() {
    SQUID_ASSERT(stream != nullptr);
    current_position = 0;
    CUDA_CHECK(cudaMemsetAsync(d_pos, 0, sizeof(int32_t), stream));
    for (int vi = 0; vi < NUM_VISITS; ++vi) {
        CUDA_CHECK(cudaMemsetAsync(k_ring[vi], 0, SCONV_RING_SLOTS * KV_DIM * 4, stream));
        CUDA_CHECK(cudaMemsetAsync(v_ring[vi], 0, SCONV_RING_SLOTS * KV_DIM * 4, stream));
        CUDA_CHECK(cudaMemsetAsync(attn_ring[vi], 0, SCONV_RING_SLOTS * HIDDEN * 4, stream));
        CUDA_CHECK(cudaMemsetAsync(mlp_ring[vi], 0, SCONV_RING_SLOTS * HIDDEN * 4, stream));
    }
    if (ds_f_ring) {
        CUDA_CHECK(cudaMemsetAsync(ds_f_ring, 0, SCONV_RING_SLOTS * HIDDEN * 4, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Engine::alloc_kv_pair(uint32_t cap_tokens, KVFormat fmt, void** k_out, void** v_out, __half** ks_out, __half** vs_out) {
    const uint64_t cap = cap_tokens;
    switch (fmt) {
        case KVFormat::BF16: {
            const uint64_t bytes = cap * KV_DIM * 2;
            *k_out = arena.alloc_raw(bytes);
            *v_out = arena.alloc_raw(bytes);
            *ks_out = nullptr;
            *vs_out = nullptr;
            break;
        }
        case KVFormat::FP8: {
            const uint64_t data_bytes = cap * KV_DIM;
            const uint64_t scale_bytes = cap * NUM_KV_HEADS * 2;
            *k_out = arena.alloc_raw(data_bytes);
            *v_out = arena.alloc_raw(data_bytes);
            *ks_out = static_cast<__half*>(arena.alloc_raw(scale_bytes));
            *vs_out = static_cast<__half*>(arena.alloc_raw(scale_bytes));
            break;
        }
        case KVFormat::Q4: {
            const uint64_t data_bytes = cap * (KV_DIM / 2);
            const uint64_t scale_bytes = cap * KV_Q4_SCALES_PER_TOK * 2;
            *k_out = arena.alloc_raw(data_bytes);
            *v_out = arena.alloc_raw(data_bytes);
            *ks_out = static_cast<__half*>(arena.alloc_raw(scale_bytes));
            *vs_out = static_cast<__half*>(arena.alloc_raw(scale_bytes));
            break;
        }
    }
}

void Engine::enable_prefix_cache() {
    SQUID_ASSERT(model != nullptr);
    if (prefix_cache.enabled) return;

    const size_t local_bytes = compute_local_state_bytes();
    const size_t shield_bytes = compute_global_kv_prefix_bytes(SIDE_SHIELD_MAX_TOKENS);
    SQUID_ASSERT(local_bytes > 0 && shield_bytes > 0);

    prefix_cache.local_state_bytes = local_bytes;
    prefix_cache.session_tokens = alloc_host_zeroed<int32_t>(context_tokens_max);
    for (size_t i = 0; i < PREFIX_CHECKPOINT_COUNT; ++i) {
        prefix_cache.checkpoints[i].data = alloc_host_zeroed<uint8_t>(local_bytes);
        prefix_cache.checkpoints[i].pos = 0;
        prefix_cache.checkpoints[i].valid = false;
    }
    prefix_cache.side_shield_buf = alloc_host_zeroed<uint8_t>(shield_bytes);
    prefix_cache.side_shield_bytes = shield_bytes;
    prefix_cache.session_len = 0;
    prefix_cache.gpu_live_at_pos = -1;
    prefix_cache.enabled = true;

    const double host_mib = static_cast<double>(
        local_bytes * PREFIX_CHECKPOINT_COUNT + shield_bytes + static_cast<size_t>(context_tokens_max) * 4) / (1024.0 * 1024.0);
    printf("[cache] Prefix KV cache enabled (%zu checkpoints + %u-tok side-request shield, host RAM=%.1f MiB, extra VRAM=0.0 MiB)\n",
           PREFIX_CHECKPOINT_COUNT, SIDE_SHIELD_MAX_TOKENS, host_mib);
    std::fflush(stdout);
}

void Engine::deinit_prefix_cache() {
    if (!prefix_cache.enabled) return;
    free_host(prefix_cache.session_tokens);
    for (size_t i = 0; i < PREFIX_CHECKPOINT_COUNT; ++i) {
        free_host(prefix_cache.checkpoints[i].data);
    }
    free_host(prefix_cache.side_shield_buf);
    prefix_cache = PrefixCache{};
}

size_t Engine::compute_local_state_bytes() const {
    size_t total = 0;
    for (int vi = 0; vi < NUM_VISITS; ++vi) {
        if (!schedule[vi].is_global) {
            total += 2 * static_cast<size_t>(kv_cap[vi]) * KV_DIM * 2;
        }
        total += 2 * static_cast<size_t>(SCONV_RING_SLOTS) * KV_DIM * 4;
        total += 2 * static_cast<size_t>(SCONV_RING_SLOTS) * HIDDEN * 4;
    }
    if (ds_f_ring != nullptr) {
        for (int ml = 0; ml < MTP_LAYERS; ++ml) {
            if (is_mtp_local_layer(ml)) {
                total += 2 * static_cast<size_t>(ds_kv_cap[ml]) * KV_DIM * 2;
            }
        }
        total += static_cast<size_t>(SCONV_RING_SLOTS) * HIDDEN * 4;
    }
    return total;
}

size_t Engine::compute_global_kv_prefix_bytes(uint32_t token_count) const {
    const GlobalKVBytes sz = global_kv_bytes_for_tokens(kv_format, token_count);
    size_t total = 0;
    for (int vi = 0; vi < NUM_VISITS; ++vi) {
        if (schedule[vi].is_global) total += 2 * sz.data + 2 * sz.scale;
    }
    if (ds_f_ring != nullptr) total += 2 * sz.data + 2 * sz.scale;
    return total;
}

void Engine::save_local_state_checkpoint(size_t slot_idx, uint32_t pos) {
    SQUID_ASSERT(prefix_cache.enabled && slot_idx < PREFIX_CHECKPOINT_COUNT);
    LocalStateCheckpoint& ckpt = prefix_cache.checkpoints[slot_idx];
    uint8_t* dst = ckpt.data;
    size_t off = 0;

    for (int vi = 0; vi < NUM_VISITS; ++vi) {
        if (!schedule[vi].is_global) {
            const size_t kv_bytes = static_cast<size_t>(kv_cap[vi]) * KV_DIM * 2;
            CUDA_CHECK(cudaMemcpyAsync(dst + off, k_cache[vi], kv_bytes, cudaMemcpyDeviceToHost, stream));
            off += kv_bytes;
            CUDA_CHECK(cudaMemcpyAsync(dst + off, v_cache[vi], kv_bytes, cudaMemcpyDeviceToHost, stream));
            off += kv_bytes;
        }
        constexpr size_t kKvRing = SCONV_RING_SLOTS * KV_DIM * 4;
        constexpr size_t kHRing = SCONV_RING_SLOTS * HIDDEN * 4;
        CUDA_CHECK(cudaMemcpyAsync(dst + off, k_ring[vi], kKvRing, cudaMemcpyDeviceToHost, stream)); off += kKvRing;
        CUDA_CHECK(cudaMemcpyAsync(dst + off, v_ring[vi], kKvRing, cudaMemcpyDeviceToHost, stream)); off += kKvRing;
        CUDA_CHECK(cudaMemcpyAsync(dst + off, attn_ring[vi], kHRing, cudaMemcpyDeviceToHost, stream)); off += kHRing;
        CUDA_CHECK(cudaMemcpyAsync(dst + off, mlp_ring[vi], kHRing, cudaMemcpyDeviceToHost, stream)); off += kHRing;
    }
    if (ds_f_ring) {
        for (int ml = 0; ml < MTP_LAYERS; ++ml) {
            if (is_mtp_local_layer(ml)) {
                const size_t kv_bytes = static_cast<size_t>(ds_kv_cap[ml]) * KV_DIM * 2;
                CUDA_CHECK(cudaMemcpyAsync(dst + off, ds_k_cache[ml], kv_bytes, cudaMemcpyDeviceToHost, stream)); off += kv_bytes;
                CUDA_CHECK(cudaMemcpyAsync(dst + off, ds_v_cache[ml], kv_bytes, cudaMemcpyDeviceToHost, stream)); off += kv_bytes;
            }
        }
        constexpr size_t kFr = SCONV_RING_SLOTS * HIDDEN * 4;
        CUDA_CHECK(cudaMemcpyAsync(dst + off, ds_f_ring, kFr, cudaMemcpyDeviceToHost, stream)); off += kFr;
    }
    SQUID_ASSERT(off == prefix_cache.local_state_bytes);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ckpt.pos = pos;
    ckpt.valid = true;
}

void Engine::restore_local_state_checkpoint(size_t slot_idx) {
    SQUID_ASSERT(prefix_cache.enabled && slot_idx < PREFIX_CHECKPOINT_COUNT);
    const LocalStateCheckpoint& ckpt = prefix_cache.checkpoints[slot_idx];
    SQUID_ASSERT(ckpt.valid);
    const uint8_t* src = ckpt.data;
    size_t off = 0;

    for (int vi = 0; vi < NUM_VISITS; ++vi) {
        if (!schedule[vi].is_global) {
            const size_t kv_bytes = static_cast<size_t>(kv_cap[vi]) * KV_DIM * 2;
            CUDA_CHECK(cudaMemcpyAsync(k_cache[vi], src + off, kv_bytes, cudaMemcpyHostToDevice, stream)); off += kv_bytes;
            CUDA_CHECK(cudaMemcpyAsync(v_cache[vi], src + off, kv_bytes, cudaMemcpyHostToDevice, stream)); off += kv_bytes;
        }
        constexpr size_t kKvRing = SCONV_RING_SLOTS * KV_DIM * 4;
        constexpr size_t kHRing = SCONV_RING_SLOTS * HIDDEN * 4;
        CUDA_CHECK(cudaMemcpyAsync(k_ring[vi], src + off, kKvRing, cudaMemcpyHostToDevice, stream)); off += kKvRing;
        CUDA_CHECK(cudaMemcpyAsync(v_ring[vi], src + off, kKvRing, cudaMemcpyHostToDevice, stream)); off += kKvRing;
        CUDA_CHECK(cudaMemcpyAsync(attn_ring[vi], src + off, kHRing, cudaMemcpyHostToDevice, stream)); off += kHRing;
        CUDA_CHECK(cudaMemcpyAsync(mlp_ring[vi], src + off, kHRing, cudaMemcpyHostToDevice, stream)); off += kHRing;
    }
    if (ds_f_ring) {
        for (int ml = 0; ml < MTP_LAYERS; ++ml) {
            if (is_mtp_local_layer(ml)) {
                const size_t kv_bytes = static_cast<size_t>(ds_kv_cap[ml]) * KV_DIM * 2;
                CUDA_CHECK(cudaMemcpyAsync(ds_k_cache[ml], src + off, kv_bytes, cudaMemcpyHostToDevice, stream)); off += kv_bytes;
                CUDA_CHECK(cudaMemcpyAsync(ds_v_cache[ml], src + off, kv_bytes, cudaMemcpyHostToDevice, stream)); off += kv_bytes;
            }
        }
        constexpr size_t kFr = SCONV_RING_SLOTS * HIDDEN * 4;
        CUDA_CHECK(cudaMemcpyAsync(ds_f_ring, src + off, kFr, cudaMemcpyHostToDevice, stream)); off += kFr;
    }
    SQUID_ASSERT(off == prefix_cache.local_state_bytes);
}

void Engine::copy_global_kv_shield(uint32_t token_count, bool save_d2h) {
    SQUID_ASSERT(prefix_cache.enabled && token_count > 0 && token_count <= SIDE_SHIELD_MAX_TOKENS);
    uint8_t* buf = prefix_cache.side_shield_buf;
    const GlobalKVBytes sz = global_kv_bytes_for_tokens(kv_format, token_count);
    size_t off = 0;

    auto xfer = [&](void* dev, size_t nbytes) {
        if (save_d2h) CUDA_CHECK(cudaMemcpyAsync(buf + off, dev, nbytes, cudaMemcpyDeviceToHost, stream));
        else          CUDA_CHECK(cudaMemcpyAsync(dev, buf + off, nbytes, cudaMemcpyHostToDevice, stream));
        off += nbytes;
    };

    for (int vi = 0; vi < NUM_VISITS; ++vi) {
        if (!schedule[vi].is_global) continue;
        xfer(k_cache[vi], sz.data);
        xfer(v_cache[vi], sz.data);
        if (sz.scale > 0) {
            xfer(k_scale[vi], sz.scale);
            xfer(v_scale[vi], sz.scale);
        }
    }
    if (ds_f_ring) {
        const uint32_t ds_tokens = std::min(token_count, static_cast<uint32_t>(ds_kv_cap[1]));
        const GlobalKVBytes dsz = global_kv_bytes_for_tokens(kv_format, ds_tokens);
        xfer(ds_k_cache[1], dsz.data);
        xfer(ds_v_cache[1], dsz.data);
        if (dsz.scale > 0) {
            xfer(ds_k_scale[1], dsz.scale);
            xfer(ds_v_scale[1], dsz.scale);
        }
    }
    SQUID_ASSERT(off <= prefix_cache.side_shield_bytes);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

GenerateStats Engine::generate(
    const int32_t* prompt, size_t prompt_len,
    uint32_t max_new_tokens, uint32_t num_drafts,
    int32_t* out_tokens, size_t out_cap,
    TokenCallbackFn on_token, void* cb_ctx) {
    return generate_with_hint(prompt, prompt_len, 0, max_new_tokens, num_drafts, out_tokens, out_cap, on_token, cb_ctx);
}

GenerateStats Engine::generate_with_hint(
    const int32_t* prompt, size_t prompt_len_in,
    uint32_t system_prefix_hint,
    uint32_t max_new_tokens_in, uint32_t num_drafts_in,
    int32_t* out_tokens, size_t out_cap,
    TokenCallbackFn on_token, void* cb_ctx) {
    SQUID_ASSERT(model != nullptr);
    GenerateStats stats{};
    stats.prompt_tokens = static_cast<uint32_t>(prompt_len_in);
    if (prompt_len_in == 0 || max_new_tokens_in == 0 || prompt_len_in >= context_tokens_max) return stats;

    uint32_t max_new_tokens = max_new_tokens_in;
    const uint32_t room = context_tokens_max - static_cast<uint32_t>(prompt_len_in);
    if (max_new_tokens > room) max_new_tokens = room;
    if (out_cap > 0 && max_new_tokens > out_cap) max_new_tokens = static_cast<uint32_t>(out_cap);

    const ModelWeights& mw = model->mw;
    const uint32_t num_drafts = std::min(num_drafts_in, static_cast<uint32_t>(MTP_MAX_DRAFTS));
    const bool use_mtp = (num_drafts > 0 && mw.dspark.loaded);
    const uint32_t prompt_len = static_cast<uint32_t>(prompt_len_in);

    const SampleConfig cfg_h{
        sample_params.temperature >= 1e-4f ? (1.0f / sample_params.temperature) : 0.0f,
        sample_params.top_p,
        sample_params.top_k,
        0,
        sample_params.seed,
    };
    CUDA_CHECK(cudaMemcpyAsync(d_sample_cfg, &cfg_h, sizeof(SampleConfig), cudaMemcpyHostToDevice, stream));

    const uint64_t prefill_start_ns = clock_monotonic_ns();
    uint32_t start_pos = 0;
    uint32_t shield_tokens = 0;

    if (prefix_cache.enabled) {
        const uint32_t max_match = std::min(prefix_cache.session_len, prompt_len > 1 ? prompt_len - 1 : 0u);
        uint32_t common = 0;
        while (common < max_match && prefix_cache.session_tokens[common] == prompt[common]) ++common;

        const uint32_t side_need = prompt_len + max_new_tokens + 16;
        if (prefix_cache.session_len >= 1024 && common < 512 && side_need <= SIDE_SHIELD_MAX_TOKENS) {
            shield_tokens = std::min(prefix_cache.session_len, side_need);
            copy_global_kv_shield(shield_tokens, true);
            reset();
        } else if (common > 0 && use_mtp == prefix_cache.session_use_mtp) {
            const uint32_t reusable_limit = common;
            bool matched_live = false;
            if (prefix_cache.gpu_live_at_pos > 0 &&
                static_cast<uint32_t>(prefix_cache.gpu_live_at_pos) <= reusable_limit) {
                start_pos = static_cast<uint32_t>(prefix_cache.gpu_live_at_pos);
                matched_live = true;
            }
            size_t best_slot = SIZE_MAX;
            for (size_t sidx = 0; sidx < PREFIX_CHECKPOINT_COUNT; ++sidx) {
                const auto& ckpt = prefix_cache.checkpoints[sidx];
                if (ckpt.valid && ckpt.pos <= reusable_limit && ckpt.pos > start_pos) {
                    start_pos = ckpt.pos;
                    best_slot = sidx;
                    matched_live = false;
                }
            }
            if (start_pos > 0) {
                if (!matched_live) restore_local_state_checkpoint(best_slot);
                current_position = static_cast<int32_t>(start_pos);
                CUDA_CHECK(cudaMemcpyAsync(d_pos, &current_position, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                stats.cached_tokens = start_pos;
            } else {
                reset();
                for (size_t s = 0; s < PREFIX_CHECKPOINT_COUNT; ++s) prefix_cache.checkpoints[s].valid = false;
            }
        } else {
            reset();
            for (size_t s = 0; s < PREFIX_CHECKPOINT_COUNT; ++s) prefix_cache.checkpoints[s].valid = false;
        }
    } else {
        reset();
    }

    struct ShieldGuard {
        Engine* eng;
        uint32_t tokens;
        ~ShieldGuard() {
            if (tokens > 0) {
                eng->copy_global_kv_shield(tokens, false);
                eng->prefix_cache.gpu_live_at_pos = -1;
            }
        }
    } shield_guard{this, shield_tokens};

    if (prefix_cache.enabled && shield_tokens == 0) {
        for (size_t s = 0; s < PREFIX_CHECKPOINT_COUNT; ++s) {
            if (prefix_cache.checkpoints[s].valid && prefix_cache.checkpoints[s].pos > start_pos) {
                prefix_cache.checkpoints[s].valid = false;
            }
        }
    }

    const uint32_t rem_tokens = prompt_len - start_pos;
    if (use_graphs && rem_tokens > static_cast<uint32_t>(MTP_BLOCK_SIZE)) {
        ensure_prefill_b8_graph(use_mtp);
    }

    uint32_t target_sys = 0;
    uint32_t target_2k = 0;
    if (prefix_cache.enabled && shield_tokens == 0) {
        if (system_prefix_hint >= 128 && system_prefix_hint > start_pos && system_prefix_hint < prompt_len) {
            target_sys = system_prefix_hint;
        }
        const uint32_t p2k = ((prompt_len - 1) / 2048) * 2048;
        if (p2k >= 2048 && p2k > start_pos && p2k < prompt_len && p2k != target_sys) {
            target_2k = p2k;
        }
    }

    struct Target { uint32_t pos; size_t slot; };
    Target targets[2] = {
        (target_sys > 0 && (target_2k == 0 || target_sys < target_2k)) ? Target{target_sys, 0} : Target{target_2k, 1},
        (target_sys > 0 && target_2k > 0) ? (target_sys < target_2k ? Target{target_2k, 1} : Target{target_sys, 0}) : Target{0, 0},
    };
    uint32_t i = start_pos;
    for (int ti = 0; ti < 2; ++ti) {
        if (targets[ti].pos > i && targets[ti].pos < prompt_len) {
            i = prefill_range(prompt, i, targets[ti].pos, use_mtp, false);
            save_local_state_checkpoint(targets[ti].slot, i);
        }
    }

    const uint32_t last_batch = prefill_final_range(prompt, i, prompt_len, use_mtp);

    int32_t first_tok = 0;
    CUDA_CHECK(cudaMemcpyAsync(&first_tok, d_next_tokens + (last_batch - 1), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_pos, &current_position, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (prefix_cache.enabled && shield_tokens == 0) {
        std::memcpy(prefix_cache.session_tokens + start_pos, prompt + start_pos, (prompt_len - start_pos) * sizeof(int32_t));
        prefix_cache.session_len = prompt_len;
        prefix_cache.session_use_mtp = use_mtp;
        prefix_cache.gpu_live_at_pos = static_cast<int32_t>(prompt_len);
        for (size_t s = 0; s < PREFIX_CHECKPOINT_COUNT; ++s) {
            if (prefix_cache.checkpoints[s].valid && prefix_cache.checkpoints[s].pos > prompt_len) {
                prefix_cache.checkpoints[s].valid = false;
            }
        }
        if (prompt_len >= 128) save_local_state_checkpoint(2, prompt_len);
    }
    stats.prefill_ms = static_cast<double>(clock_monotonic_ns() - prefill_start_ns) / 1e6;

    if (out_cap > 0) out_tokens[0] = first_tok;
    stats.generated_tokens = 1;
    stats.last_token_id = first_tok;
    if (on_token && !on_token(cb_ctx, first_tok)) return stats;
    if (first_tok == EOS_TOKEN || stats.generated_tokens >= max_new_tokens) return stats;

    const bool use_adaptive = use_mtp && use_graphs && num_drafts == static_cast<uint32_t>(MTP_MAX_DRAFTS) && spec_threshold > 0.0f;
    if (!use_mtp) {
        CUDA_CHECK(cudaMemcpyAsync(d_tokens, &first_tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        ensure_b1_graph();
    } else if (use_adaptive) {
        CUDA_CHECK(cudaMemcpyAsync(d_anchor, &first_tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        ensure_adaptive_graphs();
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_anchor, &first_tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        ensure_spec_graph(num_drafts);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // --- DECODE ---
    int32_t prev_fed_tok = first_tok;
    bool decode_stop_now = false;
    const uint64_t decode_start_ns = clock_monotonic_ns();

    if (!use_mtp) {
        while (stats.generated_tokens < max_new_tokens && static_cast<uint32_t>(current_position) < context_tokens_max) {
            if (use_graphs) {
                CUDA_CHECK(cudaGraphLaunch(b1_graph_exec, stream));
            } else {
                record_target_forward(1, false, true, false);
                launch_inc_pos_and_copy_tok(d_pos, &d_tokens[0], &d_next_tokens[0], &d_decode_result->tokens[0], stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            if (prefix_cache.enabled && shield_tokens == 0 && prefix_cache.session_len < context_tokens_max) {
                prefix_cache.session_tokens[prefix_cache.session_len++] = prev_fed_tok;
            }
            const int32_t tok = h_decode_result->tokens[0];
            prev_fed_tok = tok;
            if (out_cap > 0) out_tokens[stats.generated_tokens] = tok;
            stats.generated_tokens += 1;
            stats.last_token_id = tok;
            current_position += 1;
            if (on_token && !on_token(cb_ctx, tok)) { decode_stop_now = true; break; }
            if (tok == EOS_TOKEN) { decode_stop_now = true; break; }
        }
    } else if (use_adaptive) {
        if (use_prompt_lookup) {
            prompt_lookup.reset();
            const size_t seed_start = (prompt_len > PromptLookupIndex::kHistCap) ? (prompt_len - PromptLookupIndex::kHistCap) : 0;
            for (size_t pi = seed_start; pi < prompt_len; ++pi) prompt_lookup.push(prompt[pi]);
            prompt_lookup.push(first_tok);
        }

        bool draft_in_flight = false;
        while (stats.generated_tokens < max_new_tokens &&
               static_cast<uint32_t>(current_position) + MTP_BLOCK_SIZE <= context_tokens_max) {
            bool lookup_hit = false;
            if (use_prompt_lookup && prompt_lookup.match(h_lookup_draft->tokens)) {
                lookup_hit = true;
                CUDA_CHECK(cudaMemcpyAsync(d_lookup_draft, h_lookup_draft, sizeof(LookupDraft), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaGraphLaunch(verify_b8_graph_exec, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                draft_in_flight = false;
                stats.lookup_steps += 1;
            }
            if (!lookup_hit) {
                if (!draft_in_flight) {
                    CUDA_CHECK(cudaGraphLaunch(draft_graph_exec, stream));
                }
                CUDA_CHECK(adaptive_graph_launch(adaptive_graph, stream));
                CUDA_CHECK(cudaEventRecord(done_event, stream));
                draft_in_flight = (max_new_tokens - stats.generated_tokens > 1) &&
                                  (static_cast<uint32_t>(current_position) + MTP_BLOCK_SIZE < context_tokens_max);
                if (draft_in_flight) {
                    CUDA_CHECK(cudaGraphLaunch(draft_graph_exec, stream));
                }
                CUDA_CHECK(cudaEventSynchronize(done_event));
            }

            const int32_t n_acc_i32 = h_decode_result->accepted_count;
            const int32_t chosen_w = lookup_hit ? MTP_BLOCK_SIZE : h_decode_result->verify_width;
            SQUID_ASSERT(chosen_w == 1 || chosen_w == 2 || chosen_w == 3 || chosen_w == 4 || chosen_w == 5 || chosen_w == 8);
            SQUID_ASSERT(n_acc_i32 >= 1 && n_acc_i32 <= chosen_w);
            const uint32_t n_acc = static_cast<uint32_t>(n_acc_i32);

            if (prefix_cache.enabled && shield_tokens == 0) {
                for (uint32_t ci = 0; ci < n_acc && prefix_cache.session_len < context_tokens_max; ++ci) {
                    const int32_t fed = (ci == 0) ? prev_fed_tok : h_decode_result->tokens[ci - 1];
                    prefix_cache.session_tokens[prefix_cache.session_len++] = fed;
                }
            }
            prev_fed_tok = h_decode_result->tokens[n_acc - 1];

            stats.spec_steps += 1;
            stats.total_drafts += static_cast<uint32_t>(chosen_w - 1);
            stats.total_accepted += (n_acc - 1);
            current_position += n_acc_i32;

            for (uint32_t idx = 0; idx < n_acc && stats.generated_tokens < max_new_tokens; ++idx) {
                const int32_t tok = h_decode_result->tokens[idx];
                if (use_prompt_lookup) prompt_lookup.push(tok);
                if (out_cap > 0) out_tokens[stats.generated_tokens] = tok;
                stats.generated_tokens += 1;
                stats.last_token_id = tok;
                if (on_token && !on_token(cb_ctx, tok)) { decode_stop_now = true; break; }
                if (tok == EOS_TOKEN) { decode_stop_now = true; break; }
            }
            if (decode_stop_now) break;
        }
        if (draft_in_flight) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    } else {
        const int32_t batch = static_cast<int32_t>(num_drafts + 1);
        while (stats.generated_tokens < max_new_tokens &&
               static_cast<uint32_t>(current_position) + static_cast<uint32_t>(batch) <= context_tokens_max) {
            if (use_graphs) {
                CUDA_CHECK(cudaGraphLaunch(spec_graph_exec[num_drafts], stream));
            } else {
                const __half* ds_ks_const[MTP_LAYERS] = {ds_k_scale[0], ds_k_scale[1], ds_k_scale[2]};
                const __half* ds_vs_const[MTP_LAYERS] = {ds_v_scale[0], ds_v_scale[1], ds_v_scale[2]};
                launch_dspark_draft(
                    mw, d_anchor, d_pos, static_cast<int>(num_drafts),
                    ds_k_cache, ds_v_cache, ds_ks_const, ds_vs_const, ds_kv_cap, kv_format,
                    split_buf, ds_f_ring, ds_ws, d_drafts, nullptr, 0.0f,
                    d_sample_cfg, d_draft_probs, d_draft_q_sel, d_draft_is_delta, stream);
                launch_assemble_verify_tokens(d_anchor, d_drafts, d_tokens, static_cast<int32_t>(num_drafts), stream);
                record_target_forward(batch, true, true, true);
                launch_spec_verify(d_next_tokens, d_drafts, batch, d_decode_result->tokens, &d_decode_result->accepted_count, d_pos, d_anchor, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            const int32_t n_acc_i32 = h_decode_result->accepted_count;
            SQUID_ASSERT(n_acc_i32 >= 1 && n_acc_i32 <= batch);
            const uint32_t n_acc = static_cast<uint32_t>(n_acc_i32);

            if (prefix_cache.enabled && shield_tokens == 0) {
                for (uint32_t ci = 0; ci < n_acc && prefix_cache.session_len < context_tokens_max; ++ci) {
                    const int32_t fed = (ci == 0) ? prev_fed_tok : h_decode_result->tokens[ci - 1];
                    prefix_cache.session_tokens[prefix_cache.session_len++] = fed;
                }
            }
            prev_fed_tok = h_decode_result->tokens[n_acc - 1];

            stats.spec_steps += 1;
            stats.total_drafts += num_drafts;
            stats.total_accepted += (n_acc - 1);
            current_position += n_acc_i32;

            for (uint32_t idx = 0; idx < n_acc && stats.generated_tokens < max_new_tokens; ++idx) {
                const int32_t tok = h_decode_result->tokens[idx];
                if (out_cap > 0) out_tokens[stats.generated_tokens] = tok;
                stats.generated_tokens += 1;
                stats.last_token_id = tok;
                if (on_token && !on_token(cb_ctx, tok)) { decode_stop_now = true; break; }
                if (tok == EOS_TOKEN) { decode_stop_now = true; break; }
            }
            if (decode_stop_now) break;
        }
    }

    if (use_mtp && !decode_stop_now && stats.generated_tokens < max_new_tokens &&
        static_cast<uint32_t>(current_position) < context_tokens_max) {
        CUDA_CHECK(cudaMemcpyAsync(d_tokens, &prev_fed_tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pos, &current_position, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        while (stats.generated_tokens < max_new_tokens && static_cast<uint32_t>(current_position) < context_tokens_max) {
            record_target_forward(1, true, true, false);
            launch_inc_pos_and_copy_tok(d_pos, &d_tokens[0], &d_next_tokens[0], &d_decode_result->tokens[0], stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            if (prefix_cache.enabled && shield_tokens == 0 && prefix_cache.session_len < context_tokens_max) {
                prefix_cache.session_tokens[prefix_cache.session_len++] = prev_fed_tok;
            }
            const int32_t tok = h_decode_result->tokens[0];
            prev_fed_tok = tok;
            if (out_cap > 0) out_tokens[stats.generated_tokens] = tok;
            stats.generated_tokens += 1;
            stats.last_token_id = tok;
            current_position += 1;
            if (on_token && !on_token(cb_ctx, tok)) break;
            if (tok == EOS_TOKEN) break;
        }
    }

    stats.decode_ms = static_cast<double>(clock_monotonic_ns() - decode_start_ns) / 1e6;
    if (prefix_cache.enabled && shield_tokens == 0) {
        prefix_cache.gpu_live_at_pos = current_position;
        if (prefix_cache.session_len >= 128) save_local_state_checkpoint(3, prefix_cache.session_len);
    }
    return stats;
}

uint32_t Engine::prefill_range(const int32_t* prompt, uint32_t start_i, uint32_t end_i, bool use_mtp, bool compute_last_lm) {
    SQUID_ASSERT(end_i >= start_i);
    uint32_t i = start_i;
    while (end_i - i >= 32 && (!compute_last_lm || end_i - i > 32)) {
        const uint32_t avail = compute_last_lm ? (end_i - i - 1) : (end_i - i);
        const uint32_t chunk = std::min(PREFILL_BATCH_MAX, avail & ~7u);
        if (chunk < 32) break;
        prefill_chunk_batched(prompt, i, chunk, use_mtp);
        i += chunk;
    }
    if (use_graphs && end_i - i >= static_cast<uint32_t>(MTP_BLOCK_SIZE)) {
        cudaGraphExec_t pgraph = prefill_b8_graph_exec[use_mtp ? 1 : 0];
        CUDA_CHECK(cudaMemcpyAsync(d_pos, &current_position, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        while (i + MTP_BLOCK_SIZE <= end_i && (!compute_last_lm || i + MTP_BLOCK_SIZE < end_i)) {
            CUDA_CHECK(cudaMemcpyAsync(d_tokens, prompt + i, MTP_BLOCK_SIZE * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaGraphLaunch(pgraph, stream));
            current_position += MTP_BLOCK_SIZE;
            i += MTP_BLOCK_SIZE;
        }
    }
    while (i < end_i) {
        const uint32_t batch = std::min<uint32_t>(MTP_BLOCK_SIZE, end_i - i);
        const bool is_last = compute_last_lm && (i + batch == end_i);
        CUDA_CHECK(cudaMemcpyAsync(d_tokens, prompt + i, batch * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pos, &current_position, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        record_target_forward(static_cast<int32_t>(batch), use_mtp, is_last, false);
        current_position += static_cast<int32_t>(batch);
        i += batch;
    }
    return i;
}

uint32_t Engine::prefill_final_range(const int32_t* prompt, uint32_t start_i, uint32_t prompt_len, bool use_mtp) {
    SQUID_ASSERT(start_i < prompt_len);
    uint32_t i = start_i;
    while (prompt_len - i > 32) {
        const uint32_t avail = prompt_len - i - 1;
        const uint32_t chunk = std::min(PREFILL_BATCH_MAX, avail & ~7u);
        if (chunk < 32) break;
        prefill_chunk_batched(prompt, i, chunk, use_mtp);
        i += chunk;
    }
    if (use_graphs && prompt_len - i > static_cast<uint32_t>(MTP_BLOCK_SIZE)) {
        cudaGraphExec_t pgraph = prefill_b8_graph_exec[use_mtp ? 1 : 0];
        CUDA_CHECK(cudaMemcpyAsync(d_pos, &current_position, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        while (i + MTP_BLOCK_SIZE < prompt_len) {
            CUDA_CHECK(cudaMemcpyAsync(d_tokens, prompt + i, MTP_BLOCK_SIZE * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaGraphLaunch(pgraph, stream));
            current_position += MTP_BLOCK_SIZE;
            i += MTP_BLOCK_SIZE;
        }
    }
    uint32_t last_batch = 1;
    while (i < prompt_len) {
        const uint32_t batch = std::min<uint32_t>(MTP_BLOCK_SIZE, prompt_len - i);
        const bool is_last = (i + batch == prompt_len);
        CUDA_CHECK(cudaMemcpyAsync(d_tokens, prompt + i, batch * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pos, &current_position, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        record_target_forward(static_cast<int32_t>(batch), use_mtp, is_last, false);
        current_position += static_cast<int32_t>(batch);
        i += batch;
        last_batch = batch;
    }
    return last_batch;
}

void Engine::prefill_chunk_batched(const int32_t* prompt, uint32_t start_i, uint32_t chunk_len, bool commit_dspark) {
    SQUID_ASSERT(chunk_len >= 8 && chunk_len <= PREFILL_BATCH_MAX && (chunk_len % 8) == 0);
    const LoadedModel& lm = *model;
    const ModelWeights& mw = lm.mw;
    const int b_i32 = static_cast<int>(chunk_len);
    const size_t b_usize = chunk_len;
    const size_t n_sub = chunk_len / 8;

    int32_t h_pos_vec[PREFILL_SUB_CHUNKS_MAX];
    for (size_t s = 0; s < n_sub; ++s) {
        h_pos_vec[s] = current_position + static_cast<int32_t>(s * 8);
    }
    CUDA_CHECK(cudaMemcpyAsync(p_pos_vec, h_pos_vec, n_sub * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(p_tokens, prompt + start_i, b_usize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    launch_embed_norm(mw.embed, mw.embed_norm, p_tokens, p_hidden, b_i32, stream);

    int block_count = 0;
    bool has_prior = false;
    bool prev_mlp_pending = false;

    const bool prof_on = (std::getenv("SQUID_PROF_PREFILL") != nullptr && start_i == 0);
    uint64_t t_pre = 0, t_dense = 0, t_attn = 0, t_post = 0, t_moe = 0, t_commit = 0, t_mark = 0;
    if (prof_on) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        t_mark = clock_monotonic_ns();
    }

    for (int vi = 0; vi < NUM_VISITS; ++vi) {
        const VisitInfo& v = schedule[vi];
        const int l = v.physical_layer;
        const LayerWeights& lw = mw.layers[l];

        bool capture_prelude = false;
        if (v.is_loop && l == LOOP_START) {
            if (v.pass == 0) {
                capture_prelude = true;
            } else {
                launch_extract_block(p_blocks, 4, 2, p_saved, b_i32, stream);
                launch_rmsnorm(p_hidden, nullptr, p_prior_last, b_i32, HIDDEN, HIDDEN, RMS_EPS, stream);
                CUDA_CHECK(cudaMemcpyAsync(p_hidden, p_prelude, b_usize * HIDDEN * 4, cudaMemcpyDeviceToDevice, stream));
                block_count = 1;
                has_prior = true;
            }
        }
        const bool use_prior = v.is_loop && has_prior;
        const int tap_slot = !commit_dspark ? -1
                           : (vi == MTP_TARGET_VISITS[0]) ? 0
                           : (vi == MTP_TARGET_VISITS[1]) ? 1
                           : (vi == MTP_TARGET_VISITS[2]) ? 2 : -1;
        const bool distinct_inputs = v.is_global || use_prior;
        const bool is_block_start = (l % BLOCK_LAYERS == 0);

        if (prev_mlp_pending) {
            const LayerWeights& prev_lw = mw.layers[schedule[vi - 1].physical_layer];
            const AblationSlice abl = ablation_slice(lm, true, vi - 1);
            launch_prefill_sconv_post_res(
                p_moe_slot_out, 0, prev_lw.mlp_sconv_w, mlp_ring[vi - 1],
                p_residual, p_hidden, capture_prelude ? p_prelude : nullptr,
                nullptr, nullptr, 0, 0, b_i32, &p_pos_vec[0], abl.r, abl.d, abl.p, stream);
            if (tap_slot >= 0) {
                launch_prefill_copy_taps(p_hidden, p_taps, tap_slot, b_i32, stream);
            }
        }
        prev_mlp_pending = false;

        const __nv_bfloat16 *n0 = nullptr, *n1 = nullptr, *n2 = nullptr;
        float *o0 = nullptr, *o1 = nullptr, *o2 = nullptr;
        __half *oh0 = nullptr, *oh1 = nullptr, *oh2 = nullptr;
        int n_out = 1;

        if (!distinct_inputs) {
            n0 = lw.attn_norm;
            oh0 = p_attn_in_h[0];
            n_out = 1;
        } else if (v.is_global && !use_prior) {
            n0 = lw.q_in_norm; n1 = lw.k_in_norm; n2 = lw.v_in_norm;
            oh0 = p_attn_in_h[0]; oh1 = p_attn_in_h[1]; oh2 = p_attn_in_h[2];
            n_out = 3;
        } else if (v.is_global) {
            o0 = p_attn_in[0]; o1 = p_attn_in[1]; o2 = p_attn_in[2];
            n_out = 3;
        } else {
            o0 = p_mlp_in;
            n_out = 1;
        }

        launch_fused_depth_norm(
            p_hidden, is_block_start ? p_partial : nullptr, p_blocks, block_count,
            l > 0 ? mw.attn_queries[l - 1] : nullptr,
            n0, n1, n2, o0, o1, o2, oh0, oh1, oh2, n_out, b_i32, stream);

        if (use_prior) {
            const size_t loc = static_cast<size_t>(l - LOOP_START);
            const float* cp_q[3] = {
                mw.cross_pass_queries + (2 * loc) * HIDDEN,
                mw.cross_pass_kv_queries + (2 * loc) * HIDDEN,
                mw.cross_pass_kv_queries + (2 * loc + 1) * HIDDEN,
            };
            const __nv_bfloat16* pnorms[3] = {
                v.is_global ? lw.q_in_norm : lw.attn_norm,
                v.is_global ? lw.k_in_norm : lw.attn_norm,
                v.is_global ? lw.v_in_norm : lw.attn_norm,
            };
            for (int cs = 0; cs < 3; ++cs) {
                const float* x_in = v.is_global ? p_attn_in[cs] : p_mlp_in;
                launch_prefill_controller_one(
                    x_in, p_saved, p_prior_last, cp_q[cs], lw.ctrl[cs], pnorms[cs],
                    nullptr, p_attn_in_h[cs], p_ctrl_msg, p_ctrl_cat_h, p_ctrl_d, p_ctrl_d_h,
                    p_ctrl_f, p_ctrl_g, p_ctrl_w_f16, b_i32, stream);
            }
        }
        if (prof_on) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            uint64_t now = clock_monotonic_ns();
            t_pre += now - t_mark; t_mark = now;
        }

        const __half* in_qh = p_attn_in_h[0];
        const __half* in_kh = distinct_inputs ? p_attn_in_h[1] : p_attn_in_h[0];
        const __half* in_vh = distinct_inputs ? p_attn_in_h[2] : p_attn_in_h[0];
        launch_prefill_gemm_h(lw.w_qrgs, in_qh, p_qrgs, p_w_f16, b_i32, stream);
        launch_prefill_gemm_h(lw.w_k, in_kh, p_k_proj, p_w_f16, b_i32, stream);
        launch_prefill_gemm_h(lw.w_v, in_vh, p_v_proj, p_w_f16, b_i32, stream);
        if (prof_on) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            uint64_t now = clock_monotonic_ns();
            t_dense += now - t_mark; t_mark = now;
        }

        if (v.is_global) {
            launch_prefill_global_sconv_prep_store(
                p_qrgs, p_k_proj, p_v_proj, in_kh, in_vh,
                lw.k_sconv_w, lw.v_sconv_w, lw.k_sconv_proj, lw.v_sconv_proj,
                k_ring[vi], v_ring[vi], lw.q_clip_scale, lw.rel_clip_scale,
                k_cache[vi], v_cache[vi], k_scale[vi], v_scale[vi],
                kv_format, kv_cap[vi], p_ctrl_d, p_w_f16, b_i32, &p_pos_vec[0], stream);
            launch_prefill_flash_attn_tc(
                p_qrgs, lw.rel_logits_proj, k_cache[vi], v_cache[vi], k_scale[vi], v_scale[vi],
                kv_format, p_attn_gated_h, kv_cap[vi], 0, b_i32, &p_pos_vec[0], stream);
        } else {
            for (uint32_t off_u32 = 0; off_u32 < chunk_len; ) {
                const uint32_t sub_b = std::min(PREFILL_LOCAL_ATTN_SUB, chunk_len - off_u32);
                const size_t off = off_u32;
                const int32_t* d_pos_sub = &p_pos_vec[off / 8];
                float* qrgs_sub = p_qrgs + off * QRGS_OUT_DIM;
                float* k_sub = p_k_proj + off * KV_DIM;
                float* v_sub = p_v_proj + off * KV_DIM;

                launch_prefill_local_sconv_prep_store(
                    qrgs_sub, k_sub, v_sub, lw.k_sconv_w, lw.v_sconv_w,
                    k_ring[vi], v_ring[vi], lw.q_clip_scale, lw.rel_clip_scale,
                    static_cast<__nv_bfloat16*>(k_cache[vi]), static_cast<__nv_bfloat16*>(v_cache[vi]),
                    kv_cap[vi], static_cast<int>(sub_b), d_pos_sub, stream);
                launch_prefill_flash_attn_tc(
                    qrgs_sub, lw.rel_logits_proj, k_cache[vi], v_cache[vi], nullptr, nullptr,
                    KVFormat::BF16, p_attn_gated_h + off * Q_DIM,
                    kv_cap[vi], SWA_WINDOW, static_cast<int>(sub_b), d_pos_sub, stream);
                off_u32 += sub_b;
            }
        }
        if (prof_on) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            uint64_t now = clock_monotonic_ns();
            t_attn += now - t_mark; t_mark = now;
        }

        launch_prefill_gemm_h(lw.w_o, p_attn_gated_h, p_o_proj, p_w_f16, b_i32, stream);
        if (prof_on) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            uint64_t now = clock_monotonic_ns();
            t_dense += now - t_mark; t_mark = now;
        }

        const bool is_boundary = (l % BLOCK_LAYERS == 0);
        const AblationSlice attn_abl = ablation_slice(lm, false, vi);
        launch_prefill_sconv_post_res(
            p_o_proj, 0, lw.attn_sconv_w, attn_ring[vi], p_hidden, p_residual, nullptr,
            p_blocks, p_partial, block_count, is_boundary ? 1 : 0, b_i32, &p_pos_vec[0],
            attn_abl.r, attn_abl.d, attn_abl.p, stream);
        if (is_boundary) block_count += 1;

        launch_fused_depth_norm(
            p_residual, nullptr, p_blocks, block_count, mw.mlp_queries[l],
            use_prior ? nullptr : lw.mlp_norm, nullptr, nullptr,
            use_prior ? p_mlp_in : nullptr, nullptr, nullptr,
            use_prior ? nullptr : p_mlp_normed_h, nullptr, nullptr,
            1, b_i32, stream);

        if (use_prior) {
            const size_t loc = static_cast<size_t>(l - LOOP_START);
            launch_prefill_controller_one(
                p_mlp_in, p_saved, p_prior_last, mw.cross_pass_queries + (2 * loc + 1) * HIDDEN,
                lw.ctrl[3], lw.mlp_norm, nullptr, p_mlp_normed_h,
                p_ctrl_msg, p_ctrl_cat_h, p_ctrl_d, p_ctrl_d_h, p_ctrl_f, p_ctrl_g, p_ctrl_w_f16,
                b_i32, stream);
        }
        if (prof_on) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            uint64_t now = clock_monotonic_ns();
            t_post += now - t_mark; t_mark = now;
        }

        LinearWeight router_lw{};
        if (mw.mode == QuantMode::NVFP4) {
            router_lw = lw.router_nvfp4;
        } else {
            router_lw.kind = WeightKind::BF16;
            router_lw.rows = NUM_EXPERTS;
            router_lw.cols = HIDDEN;
            router_lw.data = lw.router_w;
        }
        launch_prefill_gemm_h(router_lw, p_mlp_normed_h, p_router_logits, p_w_f16, b_i32, stream);
        launch_prefill_route_and_sort(
            p_router_logits, lw.router_bias, p_route_ids, p_route_w,
            p_exp_offsets, p_sorted_tok, p_sorted_slot, p_sorted_w, b_i32, stream);
        launch_prefill_moe_gate_up_h(lw.w13, p_mlp_normed_h, p_exp_offsets, p_sorted_tok, p_sorted_slot, p_moe_act_h, b_i32, stream);
        launch_prefill_moe_down_h(lw.w2, p_moe_act_h, p_exp_offsets, p_sorted_tok, p_sorted_slot, p_sorted_w, p_moe_slot_out, b_i32, stream);
        if (prof_on) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            uint64_t now = clock_monotonic_ns();
            t_moe += now - t_mark; t_mark = now;
        }

        if (vi == 17 || vi == 29) {
            const AblationSlice mlp_abl = ablation_slice(lm, true, vi);
            launch_prefill_sconv_post_res(
                p_moe_slot_out, 0, lw.mlp_sconv_w, mlp_ring[vi], p_residual, p_hidden, nullptr,
                nullptr, nullptr, 0, 0, b_i32, &p_pos_vec[0], mlp_abl.r, mlp_abl.d, mlp_abl.p, stream);
            prev_mlp_pending = false;
        } else {
            prev_mlp_pending = true;
        }
    }

    const LayerWeights& last_lw = mw.layers[schedule[NUM_VISITS - 1].physical_layer];
    const AblationSlice last_mlp_abl = ablation_slice(lm, true, NUM_VISITS - 1);
    launch_prefill_sconv_post_res(
        p_moe_slot_out, 0, last_lw.mlp_sconv_w, mlp_ring[NUM_VISITS - 1], p_residual, p_hidden, nullptr,
        nullptr, nullptr, 0, 0, b_i32, &p_pos_vec[0], last_mlp_abl.r, last_mlp_abl.d, last_mlp_abl.p, stream);
    launch_fused_depth_norm(
        p_hidden, nullptr, p_blocks, block_count, mw.output_query,
        mw.final_norm, nullptr, nullptr, p_mlp_normed, nullptr, nullptr,
        nullptr, nullptr, nullptr, 1, b_i32, stream);

    if (commit_dspark) {
        launch_prefill_copy_taps(p_mlp_normed, p_taps, 3, b_i32, stream);
        launch_prefill_dspark_commit(
            mw.dspark, p_taps, ds_k_cache, ds_v_cache, ds_k_scale, ds_v_scale, ds_kv_cap,
            kv_format, ds_f_ring, p_attn_in_h[0], p_qrgs, p_w_f16, b_i32, &p_pos_vec[0], stream);
    }

    if (prof_on) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        t_commit = clock_monotonic_ns() - t_mark;
        const double tot_ms = static_cast<double>(t_pre + t_dense + t_attn + t_post + t_moe + t_commit) / 1e6;
        printf("[prof-prefill B=%u] total=%.2fms (%.0f tok/s) | pre_sub=%.2fms dense_tc=%.2fms attn_sub=%.2fms post_sub=%.2fms moe=%.2fms commit=%.2fms\n",
               chunk_len, tot_ms, static_cast<double>(chunk_len) * 1000.0 / tot_ms,
               static_cast<double>(t_pre) / 1e6, static_cast<double>(t_dense) / 1e6,
               static_cast<double>(t_attn) / 1e6, static_cast<double>(t_post) / 1e6,
               static_cast<double>(t_moe) / 1e6, static_cast<double>(t_commit) / 1e6);
        std::fflush(stdout);
    }
    current_position += b_i32;
}

double Engine::bench_step_at_pos(int32_t sim_pos_in, uint32_t num_drafts, uint32_t iterations) {
    SQUID_ASSERT(iterations > 0 && iterations <= 256);
    const ModelWeights& mw = model->mw;
    const bool use_mtp = (num_drafts > 0 && mw.dspark.loaded);
    const int32_t max_allowed = static_cast<int32_t>(context_tokens_max) - static_cast<int32_t>(iterations * MTP_BLOCK_SIZE) - 32;
    int32_t sim_pos = std::min(sim_pos_in, std::max(0, max_allowed));

    reset();

    auto zero_global_kv = [&](void* kc, void* vc, __half* ks, __half* vs, int cap_i) {
        const size_t cap = static_cast<size_t>(cap_i);
        const size_t data_bytes = (kv_format == KVFormat::BF16) ? (cap * KV_DIM * 2)
                                : (kv_format == KVFormat::FP8)  ? (cap * KV_DIM)
                                                                : (cap * (KV_DIM / 2));
        CUDA_CHECK(cudaMemsetAsync(kc, 0, data_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(vc, 0, data_bytes, stream));
        if (ks) {
            const size_t scale_bytes = (kv_format == KVFormat::FP8) ? (cap * NUM_KV_HEADS * 2)
                                                                    : (cap * KV_Q4_SCALES_PER_TOK * 2);
            CUDA_CHECK(cudaMemsetAsync(ks, 0, scale_bytes, stream));
            CUDA_CHECK(cudaMemsetAsync(vs, 0, scale_bytes, stream));
        }
    };

    for (int vi = 0; vi < NUM_VISITS; ++vi) {
        if (schedule[vi].is_global) zero_global_kv(k_cache[vi], v_cache[vi], k_scale[vi], v_scale[vi], kv_cap[vi]);
    }
    if (use_mtp) zero_global_kv(ds_k_cache[1], ds_v_cache[1], ds_k_scale[1], ds_v_scale[1], ds_kv_cap[1]);

    const int32_t seed_tok = 150000;
    CUDA_CHECK(cudaMemcpyAsync(d_tokens, &seed_tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_anchor, &seed_tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_pos, &sim_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    if (!use_mtp) ensure_b1_graph();
    else ensure_spec_graph(num_drafts);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraphExec_t exec = use_mtp ? spec_graph_exec[num_drafts] : b1_graph_exec;
    for (int w = 0; w < 4; ++w) {
        CUDA_CHECK(cudaMemcpyAsync(d_pos, &sim_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpyAsync(d_pos, &sim_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0, stream));
    for (uint32_t it = 0; it < iterations; ++it) {
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
    }
    CUDA_CHECK(cudaEventRecord(ev1, stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return static_cast<double>(elapsed_ms) / static_cast<double>(iterations);
}

bool Engine::eval_logits_streaming(
    const int32_t* tokens, size_t count,
    float* chunk_buf, size_t chunk_buf_len,
    LogitsChunkCallbackFn on_chunk, void* cb_ctx) {
    constexpr size_t kVocab = VOCAB_UNPADDED;
    SQUID_ASSERT(count > 0 && count < context_tokens_max);
    SQUID_ASSERT(chunk_buf_len >= static_cast<size_t>(MTP_BLOCK_SIZE) * kVocab);
    reset();

    const SampleConfig dump_cfg{-1.0f, 1.0f, 0, 0, 0};
    CUDA_CHECK(cudaMemcpyAsync(d_sample_cfg, &dump_cfg, sizeof(SampleConfig), cudaMemcpyHostToDevice, stream));

    struct CfgReset {
        Engine* eng;
        ~CfgReset() {
            const SampleConfig zero_cfg{0.0f, 1.0f, 0, 0, 0};
            CUDA_CHECK(cudaMemcpyAsync(eng->d_sample_cfg, &zero_cfg, sizeof(SampleConfig), cudaMemcpyHostToDevice, eng->stream));
            CUDA_CHECK(cudaStreamSynchronize(eng->stream));
        }
    } cfg_reset{this};

    for (size_t i = 0; i < count; ) {
        const size_t batch = std::min<size_t>(MTP_BLOCK_SIZE, count - i);
        CUDA_CHECK(cudaMemcpyAsync(d_tokens, tokens + i, batch * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pos, &current_position, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        record_target_forward(static_cast<int32_t>(batch), false, true, false);
        CUDA_CHECK(cudaMemcpyAsync(chunk_buf, lm_logits, batch * kVocab * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        current_position += static_cast<int32_t>(batch);
        if (!on_chunk(cb_ctx, i, chunk_buf, batch)) return false;
        i += batch;
    }
    return true;
}

void Engine::record_target_forward(int32_t batch_size, bool commit_dspark, bool compute_lm_head, bool is_verify) {
    SQUID_ASSERT(batch_size >= 1 && batch_size <= MTP_BLOCK_SIZE);
    const LoadedModel& lm = *model;
    const ModelWeights& mw = lm.mw;
    const size_t b_usize = static_cast<size_t>(batch_size);

    launch_embed_norm(mw.embed, mw.embed_norm, d_tokens, hidden, batch_size, stream);

    int block_count = 0;
    bool has_prior = false;
    bool prev_mlp_pending = false;

    for (int vi = 0; vi < NUM_VISITS; ++vi) {
        const VisitInfo& v = schedule[vi];
        const int l = v.physical_layer;
        const LayerWeights& lw = mw.layers[l];

        float* pre_tap_out = nullptr;
        if (v.is_loop && l == LOOP_START) {
            if (v.pass == 0) {
                pre_tap_out = prelude;
            } else {
                launch_extract_block(blocks, 4, 2, saved, batch_size, stream);
                launch_rmsnorm(hidden, nullptr, prior_last, batch_size, HIDDEN, HIDDEN, RMS_EPS, stream);
                CUDA_CHECK(cudaMemcpyAsync(hidden, prelude, b_usize * HIDDEN * 4, cudaMemcpyDeviceToDevice, stream));
                block_count = 1;
                has_prior = true;
            }
        }
        const bool use_prior = v.is_loop && has_prior;

        if (commit_dspark) {
            if (vi == MTP_TARGET_VISITS[0]) pre_tap_out = taps;
            else if (vi == MTP_TARGET_VISITS[1]) pre_tap_out = taps + b_usize * HIDDEN;
            else if (vi == MTP_TARGET_VISITS[2]) pre_tap_out = taps + 2 * b_usize * HIDDEN;
        }

        const bool distinct_inputs = v.is_global || use_prior;
        float* partial_dst = (l % BLOCK_LAYERS == 0) ? partial : nullptr;
        const bool use_half_dense = (lw.w_qrgs.kind != WeightKind::BF16);

        const __nv_bfloat16 *n0 = nullptr, *n1 = nullptr, *n2 = nullptr;
        float *o0 = nullptr, *o1 = nullptr, *o2 = nullptr;
        __half *oh0 = nullptr, *oh1 = nullptr, *oh2 = nullptr;
        int n_out = 1;

        if (!distinct_inputs) {
            n0 = lw.attn_norm; o0 = attn_in[0]; oh0 = use_half_dense ? attn_in_h[0] : nullptr; n_out = 1;
        } else if (v.is_global && !use_prior) {
            n0 = lw.q_in_norm; n1 = lw.k_in_norm; n2 = lw.v_in_norm;
            o0 = attn_in[0]; o1 = attn_in[1]; o2 = attn_in[2];
            oh0 = use_half_dense ? attn_in_h[0] : nullptr;
            oh1 = use_half_dense ? attn_in_h[1] : nullptr;
            oh2 = use_half_dense ? attn_in_h[2] : nullptr;
            n_out = 3;
        } else if (v.is_global) {
            o0 = attn_in[0]; o1 = attn_in[1]; o2 = attn_in[2]; n_out = 3;
        } else {
            o0 = mlp_in; n_out = 1;
        }

        if (prev_mlp_pending) {
            const LayerWeights& prev_lw = mw.layers[schedule[vi - 1].physical_layer];
            const AblationSlice prev_abl = ablation_slice(lm, true, vi - 1);
            launch_mega_mlp_post_next_depth_norm(
                moe_down, false, prev_lw.mlp_sconv_w, mlp_ring[vi - 1],
                residual, hidden, pre_tap_out, nullptr, partial_dst,
                blocks, block_count, mw.attn_queries[l - 1],
                n0, n1, n2, o0, o1, o2, oh0, oh1, oh2, n_out, batch_size, d_pos,
                prev_abl.r, prev_abl.d, prev_abl.p, stream);
            prev_mlp_pending = false;
        } else {
            launch_fused_depth_norm(
                hidden, partial_dst, blocks, block_count,
                l > 0 ? mw.attn_queries[l - 1] : nullptr,
                n0, n1, n2, o0, o1, o2, oh0, oh1, oh2, n_out, batch_size, stream);
        }

        if (use_prior) {
            const size_t loc = static_cast<size_t>(l - LOOP_START);
            float* xs[3] = {
                v.is_global ? attn_in[0] : mlp_in,
                v.is_global ? attn_in[1] : mlp_in,
                v.is_global ? attn_in[2] : mlp_in,
            };
            float* outs[3] = {attn_in[0], attn_in[1], attn_in[2]};
            __half* outs_h[3] = {attn_in_h[0], attn_in_h[1], attn_in_h[2]};
            const float* cp_q[3] = {
                mw.cross_pass_queries + (2 * loc) * HIDDEN,
                mw.cross_pass_kv_queries + (2 * loc) * HIDDEN,
                mw.cross_pass_kv_queries + (2 * loc + 1) * HIDDEN,
            };
            const ControllerWeights* cws[3] = {&lw.ctrl[0], &lw.ctrl[1], &lw.ctrl[2]};
            const __nv_bfloat16* pnorms[3] = {
                v.is_global ? lw.q_in_norm : lw.attn_norm,
                v.is_global ? lw.k_in_norm : lw.attn_norm,
                v.is_global ? lw.v_in_norm : lw.attn_norm,
            };
            if (cws[0]->down_nvfp4.data != nullptr) {
                launch_controller_nvfp4_batched(
                    xs, saved, prior_last, cp_q, cws, pnorms, outs, use_half_dense ? outs_h : nullptr,
                    3, ctrl_msg, reinterpret_cast<__half*>(ctrl_cat), reinterpret_cast<__half*>(ctrl_d),
                    batch_size, stream);
            } else {
                launch_controller_merge_batched(
                    xs, saved, prior_last, cp_q, cws, pnorms, outs, use_half_dense ? outs_h : nullptr,
                    3, ctrl_msg, ctrl_cat, ctrl_d, batch_size, stream);
            }
        }

        const float* in_q = attn_in[0];
        const float* in_k = distinct_inputs ? attn_in[1] : attn_in[0];
        const float* in_v = distinct_inputs ? attn_in[2] : attn_in[0];
        const __half* in_qh = attn_in_h[0];
        const __half* in_kh = distinct_inputs ? attn_in_h[1] : attn_in_h[0];
        const __half* in_vh = distinct_inputs ? attn_in_h[2] : attn_in_h[0];

        if (lw.w_qrgs.kind != WeightKind::BF16) {
            launch_qkvrg_fused(lw, in_qh, in_kh, in_vh, qrgs, k_proj, v_proj, batch_size, stream);
        } else {
            launch_linear(lw.w_qrgs, in_q, qrgs, batch_size, false, stream);
            launch_linear(lw.w_k, in_k, k_proj, batch_size, false, stream);
            launch_linear(lw.w_v, in_v, v_proj, batch_size, false, stream);
        }

        if (v.is_global) {
            launch_mega_global_sconv_prep_store(
                qrgs, k_proj, v_proj, in_k, in_v,
                lw.k_sconv_w, lw.v_sconv_w, lw.k_sconv_proj, lw.v_sconv_proj,
                k_ring[vi], v_ring[vi], lw.q_clip_scale, lw.rel_clip_scale,
                k_cache[vi], v_cache[vi], k_scale[vi], v_scale[vi],
                kv_format, kv_cap[vi], batch_size, d_pos, stream);
        } else {
            launch_mega_local_sconv_prep_store(
                qrgs, k_proj, v_proj, lw.k_sconv_w, lw.v_sconv_w,
                k_ring[vi], v_ring[vi], lw.q_clip_scale, lw.rel_clip_scale,
                static_cast<__nv_bfloat16*>(k_cache[vi]), static_cast<__nv_bfloat16*>(v_cache[vi]),
                kv_cap[vi], batch_size, d_pos, stream);
        }

        launch_relative_attention_gated(
            qrgs, lw.rel_logits_proj, k_cache[vi], v_cache[vi], k_scale[vi], v_scale[vi],
            v.is_global ? kv_format : KVFormat::BF16,
            split_buf,
            use_half_dense ? nullptr : attn_gated,
            use_half_dense ? attn_gated_h : nullptr,
            kv_cap[vi], v.is_global ? 0 : SWA_WINDOW, batch_size, d_pos, stream);

        if (use_half_dense) {
            launch_linear_h(lw.w_o, attn_gated_h, o_proj, batch_size, false, stream);
        } else {
            launch_linear(lw.w_o, attn_gated, o_proj, batch_size, false, stream);
        }

        const bool is_boundary = (l % BLOCK_LAYERS == 0);
        const AblationSlice attn_abl = ablation_slice(lm, false, vi);
        launch_mega_attn_post_mlp_in(
            o_proj, lw.attn_sconv_w, attn_ring[vi], hidden, partial, blocks, block_count,
            is_boundary, residual, mw.mlp_queries[l],
            use_prior ? nullptr : lw.mlp_norm,
            use_prior ? mlp_in : mlp_normed,
            use_prior ? nullptr : mlp_normed_h,
            batch_size, d_pos, attn_abl.r, attn_abl.d, attn_abl.p, stream);
        if (is_boundary) block_count += 1;

        if (use_prior) {
            const size_t loc = static_cast<size_t>(l - LOOP_START);
            float* xs1[1] = {mlp_in};
            float* outs1[1] = {mlp_normed};
            __half* outs_h1[1] = {mlp_normed_h};
            const float* cp_q1[3] = {mw.cross_pass_queries + (2 * loc + 1) * HIDDEN, nullptr, nullptr};
            const ControllerWeights* cws1[3] = {&lw.ctrl[3], nullptr, nullptr};
            const __nv_bfloat16* pnorms1[3] = {lw.mlp_norm, nullptr, nullptr};
            if (lw.ctrl[3].down_nvfp4.data != nullptr) {
                launch_controller_nvfp4_batched(
                    xs1, saved, prior_last, cp_q1, cws1, pnorms1, outs1, outs_h1,
                    1, ctrl_msg, reinterpret_cast<__half*>(ctrl_cat), reinterpret_cast<__half*>(ctrl_d),
                    batch_size, stream);
            } else {
                launch_controller_merge_batched(
                    xs1, saved, prior_last, cp_q1, cws1, pnorms1, outs1, outs_h1,
                    1, ctrl_msg, ctrl_cat, ctrl_d, batch_size, stream);
            }
        }

        if (mw.mode == QuantMode::NVFP4) {
            launch_route_topk_nvfp4(mlp_normed_h, lw.router_nvfp4, lw.router_bias, router_logits, route_ids, route_w, nullptr, nullptr, batch_size, stream);
        } else {
            launch_route_topk(mlp_normed, lw.router_w, lw.router_bias, router_logits, route_ids, route_w, nullptr, nullptr, batch_size, stream);
        }
        launch_moe_gate_up_h(lw.w13, mlp_normed_h, route_ids, nullptr, moe_act_h, batch_size, stream);
        launch_moe_down_h(lw.w2, moe_act_h, route_ids, route_w, nullptr, moe_down, batch_size, stream, true);

        if (vi == 17 || vi == 29) {
            const AblationSlice mlp_abl = ablation_slice(lm, true, vi);
            launch_mega_mlp_post_res(
                moe_down, false, lw.mlp_sconv_w, mlp_ring[vi], residual, hidden,
                batch_size, d_pos, mlp_abl.r, mlp_abl.d, mlp_abl.p, stream);
            prev_mlp_pending = false;
        } else {
            prev_mlp_pending = true;
        }
    }

    const AblationSlice last_mlp_abl = ablation_slice(lm, true, NUM_VISITS - 1);
    launch_mega_mlp_post_next_depth_norm(
        moe_down, false, mw.layers[schedule[NUM_VISITS - 1].physical_layer].mlp_sconv_w,
        mlp_ring[NUM_VISITS - 1], residual, hidden, nullptr,
        commit_dspark ? (taps + 3 * b_usize * HIDDEN) : nullptr,
        nullptr, blocks, block_count, mw.output_query,
        mw.final_norm, nullptr, nullptr, mlp_normed, nullptr, nullptr,
        nullptr, nullptr, nullptr, 1, batch_size, d_pos,
        last_mlp_abl.r, last_mlp_abl.d, last_mlp_abl.p, stream);

    if (compute_lm_head) {
        if (commit_dspark) {
            CUDA_CHECK(cudaEventRecord(fork_event, stream));
            CUDA_CHECK(cudaStreamWaitEvent(aux_stream, fork_event, 0));
            launch_dspark_commit(
                mw.dspark, taps, ds_k_cache, ds_v_cache, ds_k_scale, ds_v_scale, ds_kv_cap,
                kv_format, ds_f_ring, ds_mix, ds_kv, batch_size, d_pos, aux_stream);
            CUDA_CHECK(cudaEventRecord(join_event, aux_stream));
        }
        launch_lm_head_argmax(
            mw.unembed, mlp_normed, d_next_tokens, lm_logits, lm_blk_val, lm_blk_idx,
            d_sample_cfg, d_pos,
            is_verify ? d_drafts : nullptr,
            is_verify ? d_draft_probs : nullptr,
            is_verify ? d_draft_q_sel : nullptr,
            is_verify ? d_draft_is_delta : nullptr,
            is_verify ? mw.dspark.inv_draft_vocab_ids : nullptr,
            mw.dspark.draft_vocab_size, batch_size, stream);
        if (commit_dspark) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, join_event, 0));
        }
    } else if (commit_dspark) {
        launch_dspark_commit(
            mw.dspark, taps, ds_k_cache, ds_v_cache, ds_k_scale, ds_v_scale, ds_kv_cap,
            kv_format, ds_f_ring, ds_mix, ds_kv, batch_size, d_pos, stream);
    }
}

void Engine::ensure_b1_graph() {
    if (b1_graph_exec || !use_graphs) return;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    record_target_forward(1, false, true, false);
    launch_inc_pos_and_copy_tok(d_pos, &d_tokens[0], &d_next_tokens[0], &d_decode_result->tokens[0], stream);
    b1_graph_exec = end_capture_and_instantiate(stream);
}

void Engine::ensure_prefill_b8_graph(bool commit_dspark) {
    const int slot = commit_dspark ? 1 : 0;
    if (prefill_b8_graph_exec[slot] || !use_graphs) return;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    record_target_forward(MTP_BLOCK_SIZE, commit_dspark, false, false);
    launch_inc_pos_by(d_pos, MTP_BLOCK_SIZE, stream);
    prefill_b8_graph_exec[slot] = end_capture_and_instantiate(stream);
}

void Engine::ensure_spec_graph(uint32_t num_drafts) {
    SQUID_ASSERT(num_drafts >= 1 && num_drafts <= static_cast<uint32_t>(MTP_MAX_DRAFTS));
    if (spec_graph_exec[num_drafts] || !use_graphs) return;
    const ModelWeights& mw = model->mw;
    const int32_t batch = static_cast<int32_t>(num_drafts + 1);
    const __half* ds_ks_const[MTP_LAYERS] = {ds_k_scale[0], ds_k_scale[1], ds_k_scale[2]};
    const __half* ds_vs_const[MTP_LAYERS] = {ds_v_scale[0], ds_v_scale[1], ds_v_scale[2]};

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_dspark_draft(
        mw, d_anchor, d_pos, static_cast<int>(num_drafts),
        ds_k_cache, ds_v_cache, ds_ks_const, ds_vs_const, ds_kv_cap, kv_format,
        split_buf, ds_f_ring, ds_ws, d_drafts, nullptr, 0.0f,
        d_sample_cfg, d_draft_probs, d_draft_q_sel, d_draft_is_delta, stream);
    launch_assemble_verify_tokens(d_anchor, d_drafts, d_tokens, static_cast<int32_t>(num_drafts), stream);
    record_target_forward(batch, true, true, true);
    launch_spec_verify(d_next_tokens, d_drafts, batch, d_decode_result->tokens, &d_decode_result->accepted_count, d_pos, d_anchor, stream);
    spec_graph_exec[num_drafts] = end_capture_and_instantiate(stream);
}

void Engine::ensure_adaptive_graphs() {
    if (!use_graphs) return;
    const ModelWeights& mw = model->mw;
    const __half* ds_ks_const[MTP_LAYERS] = {ds_k_scale[0], ds_k_scale[1], ds_k_scale[2]};
    const __half* ds_vs_const[MTP_LAYERS] = {ds_v_scale[0], ds_v_scale[1], ds_v_scale[2]};

    if (draft_graph_exec && adaptive_graph_threshold != spec_threshold) {
        CUDA_CHECK(cudaGraphExecDestroy(draft_graph_exec));
        draft_graph_exec = nullptr;
    }

    if (!draft_graph_exec) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch_dspark_draft(
            mw, d_anchor, d_pos, MTP_MAX_DRAFTS,
            ds_k_cache, ds_v_cache, ds_ks_const, ds_vs_const, ds_kv_cap, kv_format,
            split_buf, ds_f_ring, ds_ws, d_drafts, d_draft_conf, spec_threshold,
            d_sample_cfg, d_draft_probs, d_draft_q_sel, d_draft_is_delta, stream);
        draft_graph_exec = end_capture_and_instantiate(stream);
    }

    if (!verify_b8_graph_exec) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch_assemble_verify_tokens(d_anchor, d_drafts, d_tokens, MTP_MAX_DRAFTS, stream);
        record_target_forward(MTP_BLOCK_SIZE, true, true, true);
        launch_spec_verify(d_next_tokens, d_drafts, MTP_BLOCK_SIZE, d_decode_result->tokens, &d_decode_result->accepted_count, d_pos, d_anchor, stream);
        verify_b8_graph_exec = end_capture_and_instantiate(stream);
    }

    if (adaptive_graph && adaptive_graph_threshold == spec_threshold) return;
    if (adaptive_graph) {
        adaptive_graph_destroy(adaptive_graph);
        adaptive_graph = nullptr;
    }

    CUDA_CHECK(adaptive_graph_create(
        d_anchor, d_drafts, d_draft_conf, spec_threshold, d_tokens, &d_decode_result->verify_width, stream, &adaptive_graph));
    static constexpr int32_t kWidths[5] = {1, 2, 3, 4, 8};
    for (int bidx = 0; bidx < 5; ++bidx) {
        CUDA_CHECK(adaptive_graph_branch_begin(adaptive_graph, bidx, stream));
        record_target_forward(kWidths[bidx], true, true, true);
        launch_spec_verify(d_next_tokens, d_drafts, kWidths[bidx], d_decode_result->tokens, &d_decode_result->accepted_count, d_pos, d_anchor, stream);
        CUDA_CHECK(adaptive_graph_branch_end(stream));
    }
    CUDA_CHECK(adaptive_graph_instantiate(adaptive_graph));
    adaptive_graph_threshold = spec_threshold;
}

} // namespace squid
