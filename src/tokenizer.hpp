#pragma once

// TigerStyle zero-malloc-per-request BPE Tokenizer + o200k_base UTF-8 Pre-Tokenizer
// + Native Inkling-2-Mini Chat Template Renderer + Iterative JSON Scanner Helpers.

#include "tiger.hpp"
#include <cstdint>
#include <string_view>

namespace squid {

constexpr uint32_t VOCAB_ID_CAPACITY = 204800;
constexpr uint32_t VOCAB_HASH_SLOTS = 524288;   // 2^19
constexpr uint32_t MERGE_HASH_SLOTS = 1048576;  // 2^20
constexpr uint32_t BYTE_ARENA_CAPACITY = 2 * 1024 * 1024;
constexpr uint32_t SPECIAL_TOKENS_MAX = 256;
constexpr uint32_t ENCODE_PIECE_MAX_BYTES = 65536;
constexpr uint32_t SCRATCH_CODEPOINTS_MAX = 262144;

static_assert((VOCAB_HASH_SLOTS & (VOCAB_HASH_SLOTS - 1)) == 0);
static_assert((MERGE_HASH_SLOTS & (MERGE_HASH_SLOTS - 1)) == 0);

struct TokenSlice {
    uint32_t offset = 0;
    uint16_t length = 0;
    uint8_t is_special = 0;
};

struct VocabHashSlot {
    uint64_t hash = 0;
    int32_t token_id = -1;
};

struct MergeHashSlot {
    static constexpr uint64_t kEmptyPairKey = UINT64_MAX;
    uint64_t pair_key = kEmptyPairKey;
    uint32_t rank = 0;
    int32_t merged_id = -1;
};

struct SpecialTokenEntry {
    uint32_t offset = 0;
    uint16_t length = 0;
    int32_t token_id = -1;
};

struct PieceCacheEntry {
    static constexpr size_t kCapacity = 4096;
    static constexpr size_t kMinInputBytes = 256;
    uint16_t generation = 0;
    uint8_t byte_count = 0;
    uint8_t token_count = 0;
    uint8_t bytes[28]{};
    int32_t tokens[8]{};
};

struct TokenizerScratch {
    uint32_t* codepoints = nullptr;
    uint32_t* byte_offsets = nullptr;
    int32_t* bpe_ids = nullptr;
    PieceCacheEntry* piece_cache = nullptr;
    uint16_t piece_cache_generation = 0;
    bool piece_cache_enabled = false;

    void init();
    void deinit();
    void begin_encode(size_t text_len);
};

struct Tokenizer {
    uint8_t* byte_arena = nullptr;
    uint32_t byte_arena_used = 0;
    TokenSlice* id_to_slice = nullptr;
    VocabHashSlot* vocab_table = nullptr;
    MergeHashSlot* merge_table = nullptr;
    int32_t byte_to_id[256]{};
    SpecialTokenEntry special_tokens[SPECIAL_TOKENS_MAX]{};
    uint32_t special_count = 0;
    uint32_t vocab_count = 0;
    uint32_t merge_count = 0;
    uint32_t max_token_id = 0;

    int32_t id_message_model = 200001;
    int32_t id_thinking = 200008;
    int32_t id_text = 200004;
    int32_t id_tool_json = -1;
    int32_t id_end_message = 200010;
    int32_t id_eos = 200006;

    bool load(std::string_view model_directory);
    void deinit();

    std::string_view decode_token_bytes(int32_t token_id, bool skip_special) const;

    size_t encode_into(
        std::string_view text,
        bool parse_special,
        TokenizerScratch& scratch,
        int32_t* output_ids,
        size_t output_cap) const;

    static void render_single_user_prompt(ByteWriter& writer, std::string_view prompt_text, std::string_view effort);
    static bool try_emit_thinking_effort(ByteWriter& writer, std::string_view effort);
    static void emit_thinking_effort(ByteWriter& writer, std::string_view effort);

private:
    uint32_t store_arena_bytes(std::string_view bytes);
    void insert_vocab(std::string_view raw, int32_t token_id);
    int32_t lookup_vocab(std::string_view raw) const;
    void insert_merge(uint64_t pair_key, uint32_t rank, int32_t merged_id);
    const MergeHashSlot* lookup_merge(uint64_t pair_key) const;
    int32_t lookup_special(std::string_view name, int32_t fallback) const;

    void encode_bpe_piece(std::string_view piece, TokenizerScratch& scratch, int32_t* output_ids, size_t output_cap, size_t& out_len) const;
    void pretokenize_and_encode(std::string_view text, TokenizerScratch& scratch, int32_t* output_ids, size_t output_cap, size_t& out_len) const;
    void pretokenize_and_encode_chunk(std::string_view text, TokenizerScratch& scratch, int32_t* output_ids, size_t output_cap, size_t& out_len) const;
};

// Iterative zero-allocation JSON scanner helpers (shared with server.cpp)
inline void skip_ws(std::string_view data, size_t& pos) {
    while (pos < data.size()) {
        const char c = data[pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos;
        else break;
    }
}

bool parse_json_string(std::string_view data, size_t& pos, char* out_buf, size_t out_cap, std::string_view* out_view);
int64_t parse_i64(std::string_view data, size_t& pos);
void skip_json_value(std::string_view data, size_t& pos);

} // namespace squid
