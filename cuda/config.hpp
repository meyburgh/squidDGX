#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace squid {

// Pinned Inkling-2-Mini text + DSpark MTP geometry constants
constexpr int HIDDEN = 2304;
constexpr int NUM_LAYERS = 24;
constexpr int NUM_VISITS = 48;
constexpr int LOOP_START = 6;
constexpr int LOOP_END = 18;
constexpr int LOOP_PASSES = 3;
constexpr int BLOCK_LAYERS = 6;
constexpr int GATE_RANK = 64;

constexpr int NUM_HEADS = 24;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int Q_DIM = NUM_HEADS * HEAD_DIM;     // 3072
constexpr int KV_DIM = NUM_KV_HEADS * HEAD_DIM; // 1024
constexpr int GQA_RATIO = NUM_HEADS / NUM_KV_HEADS; // 3

constexpr int D_REL = 16;
constexpr int R_DIM = NUM_HEADS * D_REL;        // 384
constexpr int REL_EXTENT = 1024;
constexpr int SWA_WINDOW = 1024;
constexpr int CONV_TAPS = 4;
constexpr int SCONV_RING_SLOTS = 16;

constexpr int NUM_EXPERTS = 128;
constexpr int TOP_K = 8;
constexpr int INTERMEDIATE = 1152;
constexpr int VOCAB_PADDED = 201024;
constexpr int VOCAB_UNPADDED = 200058;
constexpr int EOS_TOKEN = 200006;
constexpr int IMAGE_TOKEN_ID = 200054; // <|unused_200054|>

// Inkling-2-Mini Vision Encoder (inkling2_vision) constants
constexpr int VIS_HIDDEN = 1024;
constexpr int VIS_LAYERS = 23;
constexpr int VIS_HEADS = 16;
constexpr int VIS_HEAD_DIM = 64;
constexpr int VIS_INTERMEDIATE = 4096;
constexpr int VIS_PATCH_SIZE = 20;
constexpr int VIS_MERGE_SIZE = 2;
constexpr int VIS_TILE_SIZE = VIS_PATCH_SIZE * VIS_MERGE_SIZE; // 40
constexpr int VIS_PATCH_DIM = VIS_PATCH_SIZE * VIS_PATCH_SIZE * 3; // 1200
constexpr int VIS_POSEMB_GRID = 32;
constexpr int VIS_PROJ_HIDDEN = 4096;
constexpr float VIS_LN_EPS = 1e-5f;
constexpr float VIS_ROPE_THETA = 10000.0f;
constexpr int VIS_MAX_TOKENS_PER_IMG = 4096; // up to 16,384 ViT patches (2560x2560 native)

constexpr uint64_t VRAM_ALIGNMENT = 256;
constexpr int QRGS_OUT_DIM = Q_DIM + R_DIM + Q_DIM + NUM_HEADS; // 6552
constexpr int DSPARK_QKV_OUT_DIM = Q_DIM + 2 * KV_DIM;          // 5120

constexpr float RMS_EPS = 1e-6f;
constexpr float QK_CLIP = 22.627416997969522f;
constexpr float REL_CLIP = 8.0f;
constexpr int LOG_SCALE_FLOOR = 8192;
constexpr float LOG_SCALE_ALPHA = 0.1f;
constexpr float ROUTE_SCALE = 8.0f;
constexpr float GATE_SOFTCAP = 4.0f;
constexpr float UP_SOFTCAP = 25.0f;
constexpr float FINAL_SOFTCAP = 30.0f;
constexpr float INV_SQRT_HEAD_DIM = 0.08838834764831845f; // 1/sqrt(128)
constexpr float INV_SQRT_HIDDEN = 0.020833333333333332f;  // 1/sqrt(2304) = 1/48
constexpr float INV_SQRT_5 = 0.4472135954999579f;         // 1/sqrt(5)
constexpr float INV_SQRT_9 = 0.3333333333333333f;         // 1/sqrt(9)

// DSpark MTP constants
constexpr int MTP_LAYERS = 3;
constexpr int MTP_INTERMEDIATE = 9216;
constexpr int MTP_BLOCK_SIZE = 8; // anchor + 7 drafts
constexpr int MTP_MAX_DRAFTS = 7;
constexpr int MTP_MARKOV_RANK = 256;
constexpr float MTP_ROPE_THETA = 1000000.0f;
constexpr int MTP_TARGET_VISITS[4] = {5, 23, 41, 48};

constexpr bool is_global_layer(int physical_layer) {
    return (physical_layer + 1) % BLOCK_LAYERS == 0; // 5, 11, 17, 23
}

constexpr bool is_nvfp4_expert_layer(int physical_layer) {
    return physical_layer < 20; // 0..19 are NVFP4, 20..23 are MXFP8
}

constexpr bool is_mtp_local_layer(int l) {
    return l == 0 || l == 2;
}

enum class QuantMode : uint8_t {
    Exact = 0, // Official checkpoint: NVFP4 L0-19, MXFP8 L20-23, BF16 dense & LM head
    FP8   = 1, // Official experts + FP8 E4M3 dense attention & LM head
    NVFP4 = 2  // Aggressive all-NVFP4: L0-23 experts, dense attention, LM head, DSpark
};

inline const char* quant_mode_name(QuantMode m) {
    switch (m) {
        case QuantMode::Exact: return "exact";
        case QuantMode::FP8:   return "fp8";
        case QuantMode::NVFP4: return "nvfp4";
    }
    return "nvfp4";
}

enum class KVFormat : uint8_t {
    BF16 = 0, // 2.000 B/elem (exact parity)
    FP8  = 1, // 1.016 B/elem (E4M3 + per-head FP16 scale -> up to 448K context)
    Q4   = 2  // 0.531 B/elem (signed INT4 [-7..7] + group-64 FP16 scale -> up to 1M context!)
};

inline const char* kv_format_name(KVFormat f) {
    switch (f) {
        case KVFormat::BF16: return "bf16";
        case KVFormat::FP8:  return "fp8";
        case KVFormat::Q4:   return "q4";
    }
    return "fp8";
}

static_assert(Q_DIM == 3072, "Q_DIM must be 3072");
static_assert(KV_DIM == 1024, "KV_DIM must be 1024");
static_assert(QRGS_OUT_DIM == 6552, "QRGS_OUT_DIM must be 6552");
static_assert(HIDDEN % 32 == 0, "HIDDEN must be divisible by 32");
static_assert(INTERMEDIATE % 32 == 0, "INTERMEDIATE must be divisible by 32");
static_assert(MTP_INTERMEDIATE % 32 == 0, "MTP_INTERMEDIATE must be divisible by 32");
static_assert(LOOP_START + (LOOP_END - LOOP_START) * LOOP_PASSES + (NUM_LAYERS - LOOP_END) == NUM_VISITS,
              "visit schedule length must equal 48");
static_assert(MTP_BLOCK_SIZE == MTP_MAX_DRAFTS + 1, "MTP_BLOCK_SIZE must equal MTP_MAX_DRAFTS + 1");

// Device-resident sampling config (uploaded via H2D before generate; zero graph rebuild!)
// inv_temp == 0.0f => pure greedy argmax with zero extra memory traffic.
// pad bit 0 == 1 => force stage1 to write post-softcap logits even if slot 0 is greedy (batched multi-slot).
struct SampleConfig {
    float inv_temp;
    float top_p;
    int32_t top_k;
    uint32_t pad;
    uint64_t seed;
};

// Continuous batching & Paged Shared KV Pool constants
constexpr int MAX_BATCH_SLOTS = 8;
constexpr int MAX_DSPARK_SLOTS = 4;      // slots 0..3 can run DSpark MTP concurrently
constexpr int KV_PAGE_SIZE = 256;        // 256 tokens per global KV page (exact multiple of 32/64 tiles)
constexpr int KV_PAGE_SHIFT = 8;
constexpr int KV_PAGE_MASK = KV_PAGE_SIZE - 1;
constexpr int MAX_KV_PAGES = 4096;       // up to 1,048,576 tokens per sequence

struct BatchTokenMeta {
    int32_t slot_id;    // 0..MAX_BATCH_SLOTS-1
    int32_t pos;        // logical sequence position of this token
    int32_t local_off;  // 0..width-1 offset within this slot's tokens in the batch
    int32_t seq_idx;    // 0..num_seqs-1 index into BatchSeqMeta
};

struct BatchSeqMeta {
    int32_t slot_id;    // 0..MAX_BATCH_SLOTS-1
    int32_t pos0;       // logical start position of this seq's tokens in the batch
    int32_t batch_off;  // starting row b in [0..B-1]
    int32_t width;      // number of tokens for this seq in the batch (1..MTP_BLOCK_SIZE)
};

constexpr int KV_SPLIT_K = 64;    // 8 KV heads x 64 bulk splits = 512 CTAs
constexpr int KV_LOCAL_SPLITS = 17; // 17 x 64 = 1088 >= 1024+8 (1 single tile per CTA!)
constexpr int KV_TOTAL_SPLITS = KV_SPLIT_K + KV_LOCAL_SPLITS; // 81 slots in split_buf
constexpr size_t SPLIT_MD_ELEMS = static_cast<size_t>(NUM_HEADS) * MTP_BLOCK_SIZE * KV_TOTAL_SPLITS; // 15552
constexpr size_t SPLIT_M_OFF = 0;
constexpr size_t SPLIT_D_OFF = SPLIT_MD_ELEMS;
constexpr size_t SPLIT_O_OFF = 2 * SPLIT_MD_ELEMS;
constexpr size_t SPLIT_BUFFER_FLOATS = SPLIT_O_OFF + SPLIT_MD_ELEMS * HEAD_DIM; // 2021760 (single source of truth!)
constexpr int KV_Q4_GROUP = 64;  // 2 groups per 128-dim head
constexpr int KV_Q4_GROUPS_PER_HEAD = HEAD_DIM / KV_Q4_GROUP; // 2
constexpr int KV_Q4_SCALES_PER_TOK = NUM_KV_HEADS * KV_Q4_GROUPS_PER_HEAD; // 16

struct VisitInfo {
    int visit_idx;
    int physical_layer;
    int pass;   // 0 for prelude/coda, 0..2 for loop
    bool is_loop;
    bool is_global;
};

inline void build_visit_schedule(VisitInfo out[NUM_VISITS]) {
    int idx = 0;
    for (int l = 0; l < LOOP_START; ++l) {
        out[idx] = {idx, l, 0, false, is_global_layer(l)};
        idx++;
    }
    for (int p = 0; p < LOOP_PASSES; ++p) {
        for (int l = LOOP_START; l < LOOP_END; ++l) {
            out[idx] = {idx, l, p, true, is_global_layer(l)};
            idx++;
        }
    }
    for (int l = LOOP_END; l < NUM_LAYERS; ++l) {
        out[idx] = {idx, l, 0, false, is_global_layer(l)};
        idx++;
    }
}

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err__));                                           \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

} // namespace squid
