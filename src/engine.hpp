#pragma once

// TigerStyle C++20 Inference Engine calling Blackwell CUDA kernels directly (zero FFI).

#include "config.hpp"
#include "kernels.hpp"
#include "vision.hpp"
#include "vram_arena.hpp"
#include "weights.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

namespace squid {

using TokenCallbackFn = bool (*)(void* context, int32_t token_id);
using LogitsChunkCallbackFn = bool (*)(void* context, size_t pos_offset, const float* chunk_logits, size_t count);

struct SampleParams {
    float temperature = 0.0f;
    float top_p = 0.95f;
    int32_t top_k = 0;
    uint64_t seed = 0;
};

struct GenerateStats {
    uint32_t prompt_tokens = 0;
    uint32_t cached_tokens = 0;
    uint32_t generated_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    uint32_t spec_steps = 0;
    uint32_t lookup_steps = 0;
    uint32_t total_drafts = 0;
    uint32_t total_accepted = 0;
    int32_t last_token_id = -1;
};

// Host-side token filter for constrained decoding (response_format). While `choose` is set,
// generate() runs without speculative drafts and calls it for every token the GPU picked.
// `choose` is first called with logits == nullptr: return `proposed` to accept it, another
// token id to replace it, or kConstraintNeedLogits to be called again with that step's
// post-softcap logits row copied into `host_logits` ([VOCAB_UNPADDED]).
constexpr int32_t kConstraintNeedLogits = -2;
struct TokenConstraint {
    int32_t (*choose)(void* ctx, int32_t proposed, const float* logits) = nullptr;
    void* ctx = nullptr;
    float* host_logits = nullptr;
};

struct VisionInput {
    const float* d_embeds = nullptr;
    size_t count = 0;
};

struct PromptLookupIndex {
    static constexpr size_t kHistCap = 8192;
    static constexpr size_t kHistMask = kHistCap - 1;
    static constexpr size_t kTableCap = 16384;
    static constexpr size_t kTableMask = kTableCap - 1;
    static constexpr uint32_t kMinMatchLen = 16;

    int32_t hist[kHistCap]{};
    uint32_t count = 0;
    uint16_t gen = 0;
    uint64_t table_hash[kTableCap]{};
    uint32_t table_pos0[kTableCap]{};
    uint32_t table_pos1[kTableCap]{};
    uint16_t table_gen[kTableCap]{};

    void reset() {
        count = 0;
        gen = static_cast<uint16_t>(gen + 1);
        if (gen == 0) {
            std::memset(table_gen, 0, sizeof(table_gen));
            gen = 1;
        }
    }

    static inline uint64_t hash4(int32_t a, int32_t b, int32_t c, int32_t d) {
        uint64_t h = static_cast<uint64_t>(static_cast<int64_t>(a)) * 0x9E3779B97F4A7C15ULL;
        h = (h ^ static_cast<uint64_t>(static_cast<int64_t>(b))) * 0xBF58476D1CE4E5B9ULL;
        h = (h ^ static_cast<uint64_t>(static_cast<int64_t>(c))) * 0x94D049BB133111EBULL;
        h = (h ^ static_cast<uint64_t>(static_cast<int64_t>(d))) * 0x9E3779B97F4A7C15ULL;
        return h ^ (h >> 31);
    }

    inline int32_t at(uint32_t abs_idx) const { return hist[abs_idx & kHistMask]; }

    void push(int32_t tok) {
        const uint32_t idx = count;
        hist[idx & kHistMask] = tok;
        count = idx + 1;
        if (count >= 4) {
            const uint64_t h = hash4(at(count - 4), at(count - 3), at(count - 2), at(count - 1));
            const size_t slot = static_cast<size_t>(h & kTableMask);
            if (table_gen[slot] == gen && table_hash[slot] == h) {
                table_pos1[slot] = table_pos0[slot];
                table_pos0[slot] = count;
            } else {
                table_hash[slot] = h;
                table_pos0[slot] = count;
                table_pos1[slot] = 0;
                table_gen[slot] = gen;
            }
        }
    }

    bool match(int32_t out_drafts[MTP_MAX_DRAFTS]) const {
        const uint32_t n = count;
        if (n < kMinMatchLen + MTP_MAX_DRAFTS + 1) return false;
        const uint64_t h = hash4(at(n - 4), at(n - 3), at(n - 2), at(n - 1));
        const size_t slot = static_cast<size_t>(h & kTableMask);
        if (table_gen[slot] != gen || table_hash[slot] != h) return false;

        const uint32_t cand = (table_pos0[slot] == n) ? table_pos1[slot] : table_pos0[slot];
        if (cand < kMinMatchLen) return false;
        if (cand + MTP_MAX_DRAFTS >= n) return false;
        const uint32_t win_lo = (n > kHistCap) ? (n - static_cast<uint32_t>(kHistCap)) : 0;
        if (cand - kMinMatchLen < win_lo) return false;

        for (uint32_t k = 1; k <= kMinMatchLen; ++k) {
            if (at(cand - k) != at(n - k)) return false;
        }
        for (uint32_t j = 0; j < MTP_MAX_DRAFTS; ++j) {
            out_drafts[j] = at(cand + j);
        }
        return true;
    }
};

constexpr size_t PREFIX_CHECKPOINT_COUNT = 4;
constexpr uint32_t SIDE_SHIELD_MAX_TOKENS = 4096;
constexpr uint32_t PREFILL_BATCH_MAX = 2048;
constexpr uint32_t PREFILL_LOCAL_ATTN_SUB = 256;
constexpr uint32_t PREFILL_SUB_CHUNKS_MAX = PREFILL_BATCH_MAX / MTP_BLOCK_SIZE; // 256

struct LookupDraft {
    int32_t tokens[MTP_MAX_DRAFTS];
    int32_t is_delta;
};

struct DecodeResult {
    int32_t tokens[MTP_BLOCK_SIZE];
    int32_t accepted_count;
    int32_t verify_width;
};

struct LocalStateCheckpoint {
    uint32_t pos = 0;
    bool valid = false;
    uint8_t* data = nullptr;
};

// A conversation's prefix state parked in host RAM while another conversation uses the GPU:
// its tokens, the global-layer KV for those tokens, and its local-state checkpoints.
struct ParkedSession {
    bool valid = false;
    bool use_mtp = false;
    uint64_t last_used = 0;
    uint32_t len = 0;
    int32_t* tokens = nullptr;
    uint32_t tokens_cap = 0;
    uint8_t* global_kv = nullptr;
    size_t global_kv_cap = 0;
    LocalStateCheckpoint checkpoints[PREFIX_CHECKPOINT_COUNT]{};
};
constexpr uint32_t PARKED_SESSIONS_MAX = 8;
constexpr uint32_t PARK_MIN_TOKENS = 1024;   // shorter conversations are cheap to re-prefill

struct PrefixCache {
    bool enabled = false;
    uint32_t parked_count = 0;
    uint64_t use_clock = 0;
    ParkedSession parked[PARKED_SESSIONS_MAX]{};
    size_t local_state_bytes = 0;
    int32_t* session_tokens = nullptr;
    uint32_t session_len = 0;
    bool session_use_mtp = false;
    int32_t gpu_live_at_pos = -1;
    LocalStateCheckpoint checkpoints[PREFIX_CHECKPOINT_COUNT]{};
    uint8_t* side_shield_buf = nullptr;
    size_t side_shield_bytes = 0;
};

struct Engine {
    const LoadedModel* model = nullptr;
    uint32_t context_tokens_max = 0;
    bool use_graphs = true;
    KVFormat kv_format = KVFormat::BF16;
    int32_t current_position = 0;
    VisitInfo schedule[NUM_VISITS]{};

    VRAMArena arena{};
    cudaStream_t stream = nullptr;
    cudaStream_t aux_stream = nullptr;
    cudaEvent_t fork_event = nullptr;
    cudaEvent_t join_event = nullptr;
    cudaEvent_t done_event = nullptr;

    // Cached CUDA Graphs
    cudaGraphExec_t b1_graph_exec = nullptr;
    cudaGraphExec_t spec_graph_exec[MTP_MAX_DRAFTS + 1]{};
    cudaGraphExec_t draft_graph_exec = nullptr;
    cudaGraphExec_t verify_b8_graph_exec = nullptr;
    AdaptiveGraph* adaptive_graph = nullptr;
    float adaptive_graph_threshold = -1.0f;
    float spec_threshold = 0.75f;
    bool use_prompt_lookup = true;
    cudaGraphExec_t prefill_b8_graph_exec[2]{};

    // Pinned coherent decode results & lookup draft staging
    DecodeResult* h_decode_result = nullptr;
    DecodeResult* d_decode_result = nullptr;
    LookupDraft* h_lookup_draft = nullptr;
    LookupDraft* d_lookup_draft = nullptr;

    PrefixCache prefix_cache{};
    PromptLookupIndex prompt_lookup{};

    // Per-visit KV caches and sconv rings
    int kv_cap[NUM_VISITS]{};
    void* k_cache[NUM_VISITS]{};
    void* v_cache[NUM_VISITS]{};
    __half* k_scale[NUM_VISITS]{};
    __half* v_scale[NUM_VISITS]{};
    float* k_ring[NUM_VISITS]{};
    float* v_ring[NUM_VISITS]{};
    float* attn_ring[NUM_VISITS]{};
    float* mlp_ring[NUM_VISITS]{};

    // DSpark KV caches and f_ring
    int ds_kv_cap[MTP_LAYERS]{};
    void* ds_k_cache[MTP_LAYERS]{};
    void* ds_v_cache[MTP_LAYERS]{};
    __half* ds_k_scale[MTP_LAYERS]{};
    __half* ds_v_scale[MTP_LAYERS]{};
    float* ds_f_ring = nullptr;

    // Activation scratch (B <= 8)
    float* split_buf = nullptr;
    int32_t* d_pos = nullptr;
    int32_t* d_tokens = nullptr;
    int32_t* d_next_tokens = nullptr;
    int32_t* d_anchor = nullptr;
    int32_t* d_drafts = nullptr;
    float* d_draft_conf = nullptr;

    float* hidden = nullptr;
    float* prelude = nullptr;
    float* partial = nullptr;
    float* blocks = nullptr;
    float* saved = nullptr;
    float* prior_last = nullptr;
    float* attn_in[3]{};
    __half* attn_in_h[3]{};

    float* qrgs = nullptr;
    float* k_proj = nullptr;
    float* v_proj = nullptr;
    float* attn_gated = nullptr;
    __half* attn_gated_h = nullptr;
    float* o_proj = nullptr;
    float* residual = nullptr;
    float* mlp_in = nullptr;
    float* mlp_normed = nullptr;
    __half* mlp_normed_h = nullptr;
    float* ctrl_msg = nullptr;
    float* ctrl_cat = nullptr;
    float* ctrl_d = nullptr;
    float* router_logits = nullptr;
    int32_t* route_ids = nullptr;
    float* route_w = nullptr;
    __half* moe_act_h = nullptr;
    float* moe_down = nullptr;
    float* taps = nullptr;
    float* ds_mix = nullptr;
    float* ds_kv = nullptr;
    float* ds_ws = nullptr;
    float* lm_blk_val = nullptr;
    int32_t* lm_blk_idx = nullptr;
    float* lm_logits = nullptr;
    float* d_draft_probs = nullptr;
    float* d_draft_q_sel = nullptr;
    int32_t* d_draft_is_delta = nullptr;
    SampleConfig* d_sample_cfg = nullptr;
    SampleParams sample_params{};

    // Batched prefill scratch (B <= 2048)
    int32_t* p_tokens = nullptr;
    int32_t* p_pos_vec = nullptr;
    float* p_hidden = nullptr;
    float* p_prelude = nullptr;
    float* p_partial = nullptr;
    float* p_blocks = nullptr;
    float* p_saved = nullptr;
    float* p_prior_last = nullptr;
    float* p_attn_in[3]{};
    __half* p_attn_in_h[3]{};
    float* p_qrgs = nullptr;
    float* p_k_proj = nullptr;
    float* p_v_proj = nullptr;
    __half* p_attn_gated_h = nullptr;
    float* p_o_proj = nullptr;
    float* p_residual = nullptr;
    float* p_mlp_in = nullptr;
    float* p_mlp_normed = nullptr;
    __half* p_mlp_normed_h = nullptr;
    float* p_router_logits = nullptr;
    int32_t* p_route_ids = nullptr;
    float* p_route_w = nullptr;
    int32_t* p_exp_offsets = nullptr;
    int32_t* p_sorted_tok = nullptr;
    int32_t* p_sorted_slot = nullptr;
    float* p_sorted_w = nullptr;
    __half* p_moe_act_h = nullptr;
    float* p_moe_slot_out = nullptr;
    float* p_taps = nullptr;
    __half* p_w_f16 = nullptr;
    __half* p_ctrl_w_f16 = nullptr;
    float* p_ctrl_msg = nullptr;
    __half* p_ctrl_cat_h = nullptr;
    float* p_ctrl_d = nullptr;
    __half* p_ctrl_d_h = nullptr;
    float* p_ctrl_f = nullptr;
    float* p_ctrl_g = nullptr;

    TokenConstraint constraint{};
    using VisionInput = squid::VisionInput;
    VisionInput active_vision{};
    size_t active_vision_cursor = 0;
    int32_t* d_vis_dst = nullptr;
    int32_t* d_vis_src = nullptr;

    bool init(const LoadedModel* loaded, uint32_t max_ctx, bool graphs, KVFormat kv_fmt);
    void deinit();
    void reset();
    void enable_prefix_cache(uint32_t parked_sessions = 3);

    // Encode preprocessed images into a newly cudaMalloc'd GPU FP32 buffer [total_tokens, HIDDEN].
    // Caller must free with cudaFree(). Returns nullptr if vision weights are unavailable or empty.
    float* encode_images(const PreprocessedImage* images, size_t n_images, size_t* out_total_tokens);

    GenerateStats generate(
        const int32_t* prompt, size_t prompt_len,
        uint32_t max_new_tokens, uint32_t num_drafts,
        int32_t* out_tokens, size_t out_cap,
        TokenCallbackFn on_token = nullptr, void* cb_ctx = nullptr,
        VisionInput vision = {},
        float* out_first_logits = nullptr);

    GenerateStats generate_with_hint(
        const int32_t* prompt, size_t prompt_len,
        uint32_t system_prefix_hint,
        uint32_t max_new_tokens, uint32_t num_drafts,
        int32_t* out_tokens, size_t out_cap,
        TokenCallbackFn on_token = nullptr, void* cb_ctx = nullptr,
        VisionInput vision = {},
        float* out_first_logits = nullptr);

    double bench_step_at_pos(int32_t sim_pos, uint32_t num_drafts, uint32_t iterations);

    bool eval_logits_streaming(
        const int32_t* tokens, size_t count,
        float* chunk_buf, size_t chunk_buf_len,
        LogitsChunkCallbackFn on_chunk, void* cb_ctx);

private:
    void deinit_prefix_cache();
    size_t compute_local_state_bytes() const;
    size_t compute_global_kv_prefix_bytes(uint32_t token_count) const;
    void save_local_state_checkpoint(size_t slot_idx, uint32_t pos);
    void restore_local_state_checkpoint(size_t slot_idx);
    void copy_global_kv_shield(uint32_t token_count, bool save_d2h);
    void copy_global_kv(uint8_t* buf, size_t buf_cap, uint32_t token_count, bool save_d2h);
    void park_live_session(int32_t exclude_slot);
    void restore_parked_session(uint32_t slot);
    void select_session_for_prompt(const int32_t* prompt, uint32_t prompt_len, uint32_t max_new_tokens, bool use_mtp, bool prefix_cache_usable);
    void scatter_chunk_vision(const int32_t* host_tokens, int batch_size, float* target_hidden);

    void alloc_kv_pair(uint32_t cap_tokens, KVFormat fmt, void** k_out, void** v_out, __half** ks_out, __half** vs_out);

    uint32_t prefill_range(const int32_t* prompt, uint32_t start_i, uint32_t end_i, bool use_mtp, bool compute_last_lm);
    uint32_t prefill_final_range(const int32_t* prompt, uint32_t start_i, uint32_t prompt_len, bool use_mtp);
    void prefill_chunk_batched(const int32_t* prompt, uint32_t start_i, uint32_t chunk_len, bool commit_dspark);
    void record_target_forward(int32_t batch_size, bool commit_dspark, bool compute_lm_head, bool is_verify, const int32_t* host_tokens = nullptr);

    void ensure_b1_graph();
    void ensure_prefill_b8_graph(bool commit_dspark);
    void ensure_spec_graph(uint32_t num_drafts);
    void ensure_adaptive_graphs();
};

} // namespace squid
