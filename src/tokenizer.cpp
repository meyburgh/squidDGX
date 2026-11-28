#include "tokenizer.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace squid {

namespace {

struct ByteLevelTable {
    int16_t cp_to_byte[512];

    constexpr ByteLevelTable() : cp_to_byte{} {
        for (int i = 0; i < 512; ++i) cp_to_byte[i] = -1;
        bool direct[256]{};
        for (int b = 33; b <= 126; ++b) direct[b] = true;
        for (int b = 161; b <= 172; ++b) direct[b] = true;
        for (int b = 174; b <= 255; ++b) direct[b] = true;
        int next_cp = 256;
        for (int b = 0; b < 256; ++b) {
            int cp = direct[b] ? b : next_cp++;
            cp_to_byte[cp] = static_cast<int16_t>(b);
        }
    }
};

constexpr ByteLevelTable kByteLevelTable{};

inline uint32_t next_utf8(std::string_view s, size_t& pos) {
    const auto* p = reinterpret_cast<const uint8_t*>(s.data());
    const size_t n = s.size();
    const uint32_t c0 = p[pos++];
    if (c0 < 0x80) return c0;
    if ((c0 & 0xE0) == 0xC0 && pos < n) {
        const uint32_t c1 = p[pos++];
        return ((c0 & 0x1F) << 6) | (c1 & 0x3F);
    }
    if ((c0 & 0xF0) == 0xE0 && pos + 1 < n) {
        const uint32_t c1 = p[pos++];
        const uint32_t c2 = p[pos++];
        return ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
    }
    if ((c0 & 0xF8) == 0xF0 && pos + 2 < n) {
        const uint32_t c1 = p[pos++];
        const uint32_t c2 = p[pos++];
        const uint32_t c3 = p[pos++];
        return ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
    }
    return c0;
}

std::string_view decode_byte_level(std::string_view input, char* out_buf, size_t out_cap) {
    size_t out_len = 0;
    size_t p = 0;
    while (p < input.size()) {
        const size_t prev = p;
        const uint32_t cp = next_utf8(input, p);
        if (cp < 512 && kByteLevelTable.cp_to_byte[cp] >= 0) {
            SQUID_ASSERT(out_len < out_cap);
            out_buf[out_len++] = static_cast<char>(kByteLevelTable.cp_to_byte[cp]);
        } else {
            const size_t span_len = p - prev;
            SQUID_ASSERT(out_len + span_len <= out_cap);
            std::memcpy(out_buf + out_len, input.data() + prev, span_len);
            out_len += span_len;
        }
    }
    return std::string_view(out_buf, out_len);
}

inline bool cp_is_newline(uint32_t cp) {
    return cp == '\r' || cp == '\n';
}

inline bool cp_is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C ||
           cp == 0x0085 || cp == 0x00A0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

inline bool cp_is_digit(uint32_t cp) {
    if (cp >= '0' && cp <= '9') return true;
    if (cp < 0x80) return false;
    return (cp >= 0x0660 && cp <= 0x0669) || (cp >= 0x06F0 && cp <= 0x06F9) ||
           (cp >= 0x0966 && cp <= 0x096F) || (cp >= 0xFF10 && cp <= 0xFF19);
}

inline bool cp_is_upper(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return true;
    if (cp < 0x80) return false;
    return (cp >= 0x00C0 && cp <= 0x00D6) || (cp >= 0x00D8 && cp <= 0x00DE) ||
           (cp >= 0x0391 && cp <= 0x03A9) || (cp >= 0x0410 && cp <= 0x042F) ||
           (cp >= 0xFF21 && cp <= 0xFF3A);
}

inline bool cp_is_lower(uint32_t cp) {
    if (cp >= 'a' && cp <= 'z') return true;
    if (cp < 0x80) return false;
    return (cp >= 0x00DF && cp <= 0x00F6) || (cp >= 0x00F8 && cp <= 0x00FF) ||
           (cp >= 0x03B1 && cp <= 0x03C9) || (cp >= 0x0430 && cp <= 0x044F) ||
           (cp >= 0xFF41 && cp <= 0xFF5A);
}

inline bool cp_is_other_letter_or_mark(uint32_t cp) {
    if (cp < 0x80) return false;
    return (cp >= 0x0100 && cp <= 0x024F) || (cp >= 0x0300 && cp <= 0x036F) ||
           (cp >= 0x0590 && cp <= 0x06FF) || (cp >= 0x0900 && cp <= 0x097F) ||
           (cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x3400 && cp <= 0x9FFF) ||
           (cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0xF900 && cp <= 0xFAFF);
}

inline bool cp_is_letter(uint32_t cp) {
    return cp_is_upper(cp) || cp_is_lower(cp) || cp_is_other_letter_or_mark(cp);
}

inline bool cp_is_cat_U(uint32_t cp) {
    return cp_is_upper(cp) || cp_is_other_letter_or_mark(cp);
}

inline bool cp_is_cat_L(uint32_t cp) {
    return cp_is_lower(cp) || cp_is_other_letter_or_mark(cp);
}

inline size_t match_contraction(const uint32_t* cps, size_t n, size_t i) {
    if (i + 1 >= n || cps[i] != '\'') return 0;
    const uint32_t c1 = (cps[i + 1] >= 'A' && cps[i + 1] <= 'Z') ? (cps[i + 1] + 32) : cps[i + 1];
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
    if (i + 2 < n) {
        const uint32_t c2 = (cps[i + 2] >= 'A' && cps[i + 2] <= 'Z') ? (cps[i + 2] + 32) : cps[i + 2];
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) return 3;
    }
    return 0;
}

inline size_t match_letter_piece(const uint32_t* cps, size_t n, size_t start) {
    size_t end = start;
    if (!cp_is_newline(cps[end]) && !cp_is_letter(cps[end]) && !cp_is_digit(cps[end])) ++end;
    const size_t letters_start = end;
    while (end < n && cp_is_cat_U(cps[end])) ++end;
    while (end < n && cp_is_cat_L(cps[end])) ++end;
    if (end == letters_start) return 0;
    return end + match_contraction(cps, n, end);
}

inline int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

inline bool parse_hex4(std::string_view data, size_t pos, uint32_t* out) {
    if (pos + 4 > data.size()) return false;
    int h0 = hex_nibble(data[pos]);
    int h1 = hex_nibble(data[pos + 1]);
    int h2 = hex_nibble(data[pos + 2]);
    int h3 = hex_nibble(data[pos + 3]);
    if ((h0 | h1 | h2 | h3) < 0) return false;
    *out = static_cast<uint32_t>((h0 << 12) | (h1 << 8) | (h2 << 4) | h3);
    return true;
}

inline bool append_utf8_codepoint(char* out, size_t out_cap, size_t& out_len, uint32_t cp) {
    if (cp < 0x80) {
        if (out_len + 1 > out_cap) return false;
        out[out_len++] = static_cast<char>(cp);
    } else if (cp < 0x800) {
        if (out_len + 2 > out_cap) return false;
        out[out_len++] = static_cast<char>(0xC0 | (cp >> 6));
        out[out_len++] = static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        if (out_len + 3 > out_cap) return false;
        out[out_len++] = static_cast<char>(0xE0 | (cp >> 12));
        out[out_len++] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out[out_len++] = static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        if (out_len + 4 > out_cap) return false;
        out[out_len++] = static_cast<char>(0xF0 | (cp >> 18));
        out[out_len++] = static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out[out_len++] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out[out_len++] = static_cast<char>(0x80 | (cp & 0x3F));
    }
    return true;
}

void skip_json_string(std::string_view data, size_t& pos) {
    ++pos;
    while (pos < data.size()) {
        pos += json_plain_prefix<false>(data.substr(pos));
        if (pos == data.size()) return;
        if (data[pos] == '"') {
            ++pos;
            return;
        }
        pos = std::min(pos + 2, data.size());
    }
}

inline uint64_t fast_piece_hash(std::string_view s) {
    // Fast word-at-a-time mixer for <=28-byte pieces
    const auto* p = reinterpret_cast<const uint8_t*>(s.data());
    size_t n = s.size();
    uint64_t h = static_cast<uint64_t>(n) * 0x9e3779b97f4a7c15ULL;
    size_t i = 0;
    while (i + 8 <= n) {
        uint64_t w = 0;
        std::memcpy(&w, p + i, 8);
        h = (h ^ w) * 0xbf58476d1ce4e5b9ULL;
        i += 8;
    }
    if (i < n) {
        uint64_t tail = 0;
        std::memcpy(&tail, p + i, n - i);
        h = (h ^ tail) * 0x94d049bb133111ebULL;
    }
    return h ^ (h >> 31);
}

} // namespace

void TokenizerScratch::init() {
    codepoints = alloc_host_zeroed<uint32_t>(SCRATCH_CODEPOINTS_MAX);
    byte_offsets = alloc_host_zeroed<uint32_t>(SCRATCH_CODEPOINTS_MAX + 1);
    bpe_ids = alloc_host_zeroed<int32_t>(ENCODE_PIECE_MAX_BYTES);
    piece_cache = alloc_host_zeroed<PieceCacheEntry>(PieceCacheEntry::kCapacity);
    piece_cache_generation = 0;
    piece_cache_enabled = false;
}

void TokenizerScratch::deinit() {
    free_host(piece_cache);
    free_host(codepoints);
    free_host(byte_offsets);
    free_host(bpe_ids);
    *this = TokenizerScratch{};
}

void TokenizerScratch::begin_encode(size_t text_len) {
    piece_cache_enabled = text_len >= PieceCacheEntry::kMinInputBytes;
    if (!piece_cache_enabled) return;
    piece_cache_generation = static_cast<uint16_t>(piece_cache_generation + 1);
    if (piece_cache_generation == 0) {
        for (size_t i = 0; i < PieceCacheEntry::kCapacity; ++i) piece_cache[i].generation = 0;
        piece_cache_generation = 1;
    }
}

bool Tokenizer::load(std::string_view model_directory) {
    SQUID_ASSERT(!model_directory.empty());
    SQUID_ASSERT(byte_arena == nullptr);

    const uint64_t start_ns = clock_monotonic_ns();
    char path_buf[1024];
    const char* path_z = format_path_z(path_buf, sizeof(path_buf), model_directory, "/tokenizer.json");

    int fd = ::open(path_z, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[tokenizer] Failed to open %.*s/tokenizer.json\n",
                static_cast<int>(model_directory.size()), model_directory.data());
        return false;
    }

    off_t sz = ::lseek(fd, 0, SEEK_END);
    if (sz <= 2) {
        ::close(fd);
        return false;
    }
    const size_t file_bytes = static_cast<size_t>(sz);
    void* mapped_raw = ::mmap(nullptr, file_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_raw == MAP_FAILED) {
        ::close(fd);
        return false;
    }

    const std::string_view data(static_cast<const char*>(mapped_raw), file_bytes);

    byte_arena = alloc_host_zeroed<uint8_t>(BYTE_ARENA_CAPACITY);
    id_to_slice = alloc_host_zeroed<TokenSlice>(VOCAB_ID_CAPACITY);
    vocab_table = alloc_host_zeroed<VocabHashSlot>(VOCAB_HASH_SLOTS);
    merge_table = alloc_host_zeroed<MergeHashSlot>(MERGE_HASH_SLOTS);
    for (uint32_t i = 0; i < MERGE_HASH_SLOTS; ++i) {
        merge_table[i].pair_key = MergeHashSlot::kEmptyPairKey;
        merge_table[i].merged_id = -1;
    }

    char unescape_a[4096];
    char unescape_b[4096];
    char raw_a[4096];
    char raw_b[4096];
    char concat_buf[8192];

    size_t merges_offset = SIZE_MAX;
    size_t pos = 0;
    skip_ws(data, pos);
    bool ok = (pos < data.size() && data[pos] == '{');
    if (ok) ++pos;

    for (uint32_t top_iter = 0; ok && pos < data.size() && top_iter < 64; ++top_iter) {
        skip_ws(data, pos);
        if (pos < data.size() && data[pos] == '}') {
            ++pos;
            break;
        }
        std::string_view key;
        if (!parse_json_string(data, pos, unescape_a, sizeof(unescape_a), &key)) { ok = false; break; }
        skip_ws(data, pos);
        if (pos >= data.size() || data[pos] != ':') { ok = false; break; }
        ++pos;
        skip_ws(data, pos);

        if (key == "added_tokens" && pos < data.size() && data[pos] == '[') {
            ++pos;
            for (uint32_t added_iter = 0; ok && pos < data.size() && added_iter < SPECIAL_TOKENS_MAX * 4; ++added_iter) {
                skip_ws(data, pos);
                if (pos >= data.size()) break;
                if (data[pos] == ']') { ++pos; break; }
                if (data[pos] == '{') {
                    ++pos;
                    int32_t aid = -1;
                    size_t content_len = 0;
                    for (uint32_t fiter = 0; ok && pos < data.size() && fiter < 32; ++fiter) {
                        skip_ws(data, pos);
                        if (pos < data.size() && data[pos] == '}') { ++pos; break; }
                        std::string_view fk;
                        if (!parse_json_string(data, pos, unescape_a, sizeof(unescape_a), &fk)) { ok = false; break; }
                        skip_ws(data, pos);
                        if (pos < data.size() && data[pos] == ':') ++pos;
                        skip_ws(data, pos);
                        if (fk == "id") {
                            aid = static_cast<int32_t>(parse_i64(data, pos));
                        } else if (fk == "content") {
                            std::string_view content;
                            if (!parse_json_string(data, pos, unescape_b, sizeof(unescape_b), &content)) { ok = false; break; }
                            content_len = content.size();
                        } else {
                            skip_json_value(data, pos);
                        }
                        skip_ws(data, pos);
                        if (pos < data.size() && data[pos] == ',') ++pos;
                    }
                    if (ok && aid >= 0 && content_len > 0) {
                        const uint32_t uid = static_cast<uint32_t>(aid);
                        SQUID_ASSERT(uid < VOCAB_ID_CAPACITY);
                        const uint32_t off = store_arena_bytes(std::string_view(unescape_b, content_len));
                        id_to_slice[uid] = {off, static_cast<uint16_t>(content_len), 1};
                        if (uid > max_token_id) max_token_id = uid;
                        SQUID_ASSERT(special_count < SPECIAL_TOKENS_MAX);
                        special_tokens[special_count++] = {off, static_cast<uint16_t>(content_len), aid};
                    }
                } else {
                    skip_json_value(data, pos);
                }
                skip_ws(data, pos);
                if (pos < data.size() && data[pos] == ',') ++pos;
            }
        } else if (key == "model" && pos < data.size() && data[pos] == '{') {
            ++pos;
            for (uint32_t miter = 0; ok && pos < data.size() && miter < 64; ++miter) {
                skip_ws(data, pos);
                if (pos < data.size() && data[pos] == '}') { ++pos; break; }
                std::string_view mk;
                if (!parse_json_string(data, pos, unescape_a, sizeof(unescape_a), &mk)) { ok = false; break; }
                skip_ws(data, pos);
                if (pos >= data.size() || data[pos] != ':') { ok = false; break; }
                ++pos;
                skip_ws(data, pos);

                if (mk == "vocab" && pos < data.size() && data[pos] == '{') {
                    ++pos;
                    for (uint32_t viter = 0; ok && pos < data.size() && viter < VOCAB_ID_CAPACITY; ++viter) {
                        skip_ws(data, pos);
                        if (pos < data.size() && data[pos] == '}') { ++pos; break; }
                        std::string_view tok_str;
                        if (!parse_json_string(data, pos, unescape_a, sizeof(unescape_a), &tok_str)) { ok = false; break; }
                        skip_ws(data, pos);
                        if (pos < data.size() && data[pos] == ':') ++pos;
                        const int32_t id = static_cast<int32_t>(parse_i64(data, pos));
                        SQUID_ASSERT(id >= 0);
                        const uint32_t uid = static_cast<uint32_t>(id);
                        SQUID_ASSERT(uid < VOCAB_ID_CAPACITY);

                        const std::string_view raw = decode_byte_level(tok_str, raw_a, sizeof(raw_a));
                        const uint32_t off = store_arena_bytes(raw);
                        id_to_slice[uid] = {off, static_cast<uint16_t>(raw.size()), 0};
                        if (uid > max_token_id) max_token_id = uid;
                        insert_vocab(raw, id);
                        vocab_count += 1;

                        skip_ws(data, pos);
                        if (pos < data.size() && data[pos] == ',') ++pos;
                    }
                } else if (mk == "merges" && pos < data.size() && data[pos] == '[') {
                    merges_offset = pos;
                    skip_json_value(data, pos);
                } else {
                    skip_json_value(data, pos);
                }
                skip_ws(data, pos);
                if (pos < data.size() && data[pos] == ',') ++pos;
            }
        } else {
            skip_json_value(data, pos);
        }
        skip_ws(data, pos);
        if (pos < data.size() && data[pos] == ',') ++pos;
    }

    if (ok && merges_offset != SIZE_MAX) {
        size_t mp = merges_offset + 1;
        uint32_t rank = 0;
        for (uint32_t miter = 0; ok && mp < data.size() && miter < 600000; ++miter) {
            skip_ws(data, mp);
            if (mp >= data.size() || data[mp] == ']') break;

            std::string_view a_slice, b_slice;
            if (data[mp] == '[') {
                ++mp;
                skip_ws(data, mp);
                if (!parse_json_string(data, mp, unescape_a, sizeof(unescape_a), &a_slice)) { ok = false; break; }
                skip_ws(data, mp);
                if (mp < data.size() && data[mp] == ',') ++mp;
                skip_ws(data, mp);
                if (!parse_json_string(data, mp, unescape_b, sizeof(unescape_b), &b_slice)) { ok = false; break; }
                skip_ws(data, mp);
                if (mp < data.size() && data[mp] == ']') ++mp;
            } else if (data[mp] == '"') {
                std::string_view pair;
                if (!parse_json_string(data, mp, unescape_a, sizeof(unescape_a), &pair)) { ok = false; break; }
                const size_t sp = pair.find(' ');
                if (sp != std::string_view::npos) {
                    a_slice = pair.substr(0, sp);
                    b_slice = pair.substr(sp + 1);
                } else {
                    rank += 1;
                    skip_ws(data, mp);
                    if (mp < data.size() && data[mp] == ',') ++mp;
                    continue;
                }
            } else {
                skip_json_value(data, mp);
                skip_ws(data, mp);
                if (mp < data.size() && data[mp] == ',') ++mp;
                continue;
            }

            const std::string_view ra = decode_byte_level(a_slice, raw_a, sizeof(raw_a));
            const std::string_view rb = decode_byte_level(b_slice, raw_b, sizeof(raw_b));
            const int32_t id_a = lookup_vocab(ra);
            const int32_t id_b = lookup_vocab(rb);
            if (id_a >= 0 && id_b >= 0 && ra.size() + rb.size() <= sizeof(concat_buf)) {
                std::memcpy(concat_buf, ra.data(), ra.size());
                std::memcpy(concat_buf + ra.size(), rb.data(), rb.size());
                const int32_t id_ab = lookup_vocab(std::string_view(concat_buf, ra.size() + rb.size()));
                if (id_ab >= 0) {
                    const uint64_t pair_key = (static_cast<uint64_t>(static_cast<uint32_t>(id_a)) << 32) |
                                               static_cast<uint64_t>(static_cast<uint32_t>(id_b));
                    insert_merge(pair_key, rank, id_ab);
                    merge_count += 1;
                }
            }
            rank += 1;
            skip_ws(data, mp);
            if (mp < data.size() && data[mp] == ',') ++mp;
        }
    }

    ::munmap(mapped_raw, file_bytes);
    ::close(fd);

    if (!ok) {
        deinit();
        return false;
    }

    for (uint32_t b = 0; b < 256; ++b) {
        const char single = static_cast<char>(b);
        const int32_t found = lookup_vocab(std::string_view(&single, 1));
        byte_to_id[b] = found >= 0 ? found : 0;
    }

    for (uint32_t si = 1; si < special_count; ++si) {
        const SpecialTokenEntry cur = special_tokens[si];
        uint32_t sj = si;
        while (sj > 0 && special_tokens[sj - 1].length < cur.length) {
            special_tokens[sj] = special_tokens[sj - 1];
            --sj;
        }
        special_tokens[sj] = cur;
    }

    id_message_model = lookup_special("<|message_model|>", 200001);
    id_thinking = lookup_special("<|content_thinking|>", 200008);
    id_text = lookup_special("<|content_text|>", 200004);
    id_tool_json = lookup_special("<|content_invoke_tool_json|>", -1);
    id_end_message = lookup_special("<|end_message|>", 200010);
    id_eos = lookup_special("<|content_model_end_sampling|>", 200006);

    const double elapsed_ms = static_cast<double>(clock_monotonic_ns() - start_ns) / 1e6;
    printf("[tokenizer] Loaded %u vocab + %u merges + %u special tokens in %.1fms (arena=%.2f MiB)\n",
           vocab_count, merge_count, special_count, elapsed_ms,
           static_cast<double>(byte_arena_used) / (1024.0 * 1024.0));
    std::fflush(stdout);
    SQUID_ASSERT(vocab_count > 0);
    return true;
}

void Tokenizer::deinit() {
    free_host(byte_arena);
    free_host(id_to_slice);
    free_host(vocab_table);
    free_host(merge_table);
    *this = Tokenizer{};
}

std::string_view Tokenizer::decode_token_bytes(int32_t token_id, bool skip_special) const {
    SQUID_ASSERT(byte_arena != nullptr);
    if (token_id < 0) return {};
    const uint32_t uid = static_cast<uint32_t>(token_id);
    if (uid > max_token_id) return {};
    const TokenSlice& entry = id_to_slice[uid];
    if (skip_special && entry.is_special != 0) return {};
    return std::string_view(reinterpret_cast<const char*>(byte_arena + entry.offset), entry.length);
}

size_t Tokenizer::encode_into(
    std::string_view text,
    bool parse_special,
    TokenizerScratch& scratch,
    int32_t* output_ids,
    size_t output_cap) const {
    SQUID_ASSERT(byte_arena != nullptr);
    if (text.empty() || output_cap == 0) return 0;
    scratch.begin_encode(text.size());
    size_t out_len = 0;
    if (!parse_special || special_count == 0) {
        pretokenize_and_encode(text, scratch, output_ids, output_cap, out_len);
        return out_len;
    }

    size_t pos = 0;
    size_t seg_start = 0;
    const size_t iter_max = text.size() + 1;
    for (size_t iter = 0; pos < text.size() && out_len < output_cap && iter < iter_max; ++iter) {
        const size_t cand = text.find("<|", pos);
        if (cand == std::string_view::npos) break;
        int32_t matched_id = -1;
        size_t matched_len = 0;
        for (uint32_t si = 0; si < special_count; ++si) {
            const SpecialTokenEntry& sp = special_tokens[si];
            const std::string_view sp_bytes(reinterpret_cast<const char*>(byte_arena + sp.offset), sp.length);
            if (cand + sp_bytes.size() <= text.size() && text.substr(cand, sp_bytes.size()) == sp_bytes) {
                matched_id = sp.token_id;
                matched_len = sp_bytes.size();
                break;
            }
        }
        if (matched_id >= 0) {
            if (cand > seg_start) {
                pretokenize_and_encode(text.substr(seg_start, cand - seg_start), scratch, output_ids, output_cap, out_len);
            }
            if (out_len >= output_cap) return out_len;
            output_ids[out_len++] = matched_id;
            pos = cand + matched_len;
            seg_start = pos;
        } else {
            pos = cand + 2;
        }
    }
    if (seg_start < text.size() && out_len < output_cap) {
        pretokenize_and_encode(text.substr(seg_start), scratch, output_ids, output_cap, out_len);
    }
    return out_len;
}

void Tokenizer::render_single_user_prompt(ByteWriter& writer, std::string_view prompt_text, std::string_view effort) {
    SQUID_ASSERT(writer.capacity > 0);
    emit_thinking_effort(writer, effort);
    writer.append_slice("<|message_user|><|content_text|>");
    writer.append_slice(prompt_text);
    writer.append_slice("<|end_message|>");
}

bool Tokenizer::try_emit_thinking_effort(ByteWriter& writer, std::string_view effort) {
    double num = 0.0;
    if (!effort.empty()) {
        if (effort == "none") num = 0.0;
        else if (effort == "minimal") num = 0.1;
        else if (effort == "low") num = 0.2;
        else if (effort == "medium") num = 0.7;
        else if (effort == "high") num = 0.9;
        else if (effort == "xhigh" || effort == "max") num = 0.99;
        else {
            char tmp[64];
            if (effort.size() < sizeof(tmp)) {
                std::memcpy(tmp, effort.data(), effort.size());
                tmp[effort.size()] = '\0';
                char* endp = nullptr;
                double parsed = std::strtod(tmp, &endp);
                num = (endp != tmp && std::isfinite(parsed)) ? parsed : 0.0;
            }
        }
    }
    if (!std::isfinite(num) || num < 0.0) num = 0.0;
    if (num > 0.99) num = 0.99;
    if (!writer.try_append_slice("<|message_system|><|content_text|>Thinking effort level: ")) return false;
    if (num == 0.0) {
        if (!writer.try_append_slice("0")) return false;
    } else {
        char tmp[32];
        int len = std::snprintf(tmp, sizeof(tmp), "%.6f", num);
        if (len <= 0) return false;
        size_t end = static_cast<size_t>(len);
        if (std::memchr(tmp, '.', end) != nullptr) {
            while (end > 1 && tmp[end - 1] == '0') --end;
            if (end > 1 && tmp[end - 1] == '.') --end;
        }
        if (!writer.try_append_slice(std::string_view(tmp, end))) return false;
    }
    return writer.try_append_slice("<|end_message|>");
}

void Tokenizer::emit_thinking_effort(ByteWriter& writer, std::string_view effort) {
    SQUID_ASSERT(try_emit_thinking_effort(writer, effort));
}

uint32_t Tokenizer::store_arena_bytes(std::string_view bytes) {
    SQUID_ASSERT(bytes.size() <= UINT16_MAX);
    SQUID_ASSERT(byte_arena_used + bytes.size() <= BYTE_ARENA_CAPACITY);
    const uint32_t off = byte_arena_used;
    if (!bytes.empty()) std::memcpy(byte_arena + off, bytes.data(), bytes.size());
    byte_arena_used += static_cast<uint32_t>(bytes.size());
    return off;
}

void Tokenizer::insert_vocab(std::string_view raw, int32_t token_id) {
    SQUID_ASSERT(token_id >= 0);
    const uint64_t hash = hash_bytes_u64(raw);
    const uint32_t mask = VOCAB_HASH_SLOTS - 1;
    uint32_t slot = static_cast<uint32_t>(hash & mask);
    for (uint32_t probe = 0; probe < VOCAB_HASH_SLOTS; ++probe) {
        VocabHashSlot& entry = vocab_table[slot];
        if (entry.hash == 0) {
            entry.hash = hash;
            entry.token_id = token_id;
            return;
        }
        if (entry.hash == hash) {
            const TokenSlice& ex = id_to_slice[entry.token_id];
            std::string_view ex_bytes(reinterpret_cast<const char*>(byte_arena + ex.offset), ex.length);
            if (ex_bytes == raw) {
                entry.token_id = token_id;
                return;
            }
        }
        slot = (slot + 1) & mask;
    }
    SQUID_ASSERT(false);
}

int32_t Tokenizer::lookup_vocab(std::string_view raw) const {
    const uint64_t hash = hash_bytes_u64(raw);
    const uint32_t mask = VOCAB_HASH_SLOTS - 1;
    uint32_t slot = static_cast<uint32_t>(hash & mask);
    for (uint32_t probe = 0; probe < VOCAB_HASH_SLOTS; ++probe) {
        const VocabHashSlot& entry = vocab_table[slot];
        if (entry.hash == 0) return -1;
        if (entry.hash == hash) {
            const TokenSlice& ex = id_to_slice[entry.token_id];
            std::string_view ex_bytes(reinterpret_cast<const char*>(byte_arena + ex.offset), ex.length);
            if (ex_bytes == raw) return entry.token_id;
        }
        slot = (slot + 1) & mask;
    }
    return -1;
}

void Tokenizer::insert_merge(uint64_t pair_key, uint32_t rank, int32_t merged_id) {
    SQUID_ASSERT(pair_key != MergeHashSlot::kEmptyPairKey);
    SQUID_ASSERT(merged_id >= 0);
    const uint32_t mask = MERGE_HASH_SLOTS - 1;
    uint32_t slot = static_cast<uint32_t>(hash_pair_u64(pair_key) & mask);
    for (uint32_t probe = 0; probe < MERGE_HASH_SLOTS; ++probe) {
        MergeHashSlot& entry = merge_table[slot];
        if (entry.pair_key == MergeHashSlot::kEmptyPairKey) {
            entry.pair_key = pair_key;
            entry.rank = rank;
            entry.merged_id = merged_id;
            return;
        }
        if (entry.pair_key == pair_key) return;
        slot = (slot + 1) & mask;
    }
    SQUID_ASSERT(false);
}

inline const MergeHashSlot* Tokenizer::lookup_merge(uint64_t pair_key) const {
    const uint32_t mask = MERGE_HASH_SLOTS - 1;
    uint32_t slot = static_cast<uint32_t>(hash_pair_u64(pair_key) & mask);
    for (uint32_t probe = 0; probe < MERGE_HASH_SLOTS; ++probe) {
        const MergeHashSlot& entry = merge_table[slot];
        if (entry.pair_key == MergeHashSlot::kEmptyPairKey) return nullptr;
        if (entry.pair_key == pair_key) return &entry;
        slot = (slot + 1) & mask;
    }
    return nullptr;
}

int32_t Tokenizer::lookup_special(std::string_view name, int32_t fallback) const {
    for (uint32_t si = 0; si < special_count; ++si) {
        const SpecialTokenEntry& sp = special_tokens[si];
        std::string_view bytes(reinterpret_cast<const char*>(byte_arena + sp.offset), sp.length);
        if (bytes == name) return sp.token_id;
    }
    return fallback;
}

void Tokenizer::encode_bpe_piece(
    std::string_view piece,
    TokenizerScratch& scratch,
    int32_t* output_ids,
    size_t output_cap,
    size_t& out_len) const {
    if (piece.empty() || out_len >= output_cap) return;
    const auto* p = reinterpret_cast<const uint8_t*>(piece.data());
    if (piece.size() == 1) {
        output_ids[out_len++] = byte_to_id[p[0]];
        return;
    }
    if (piece.size() == 2) {
        const int32_t id0 = byte_to_id[p[0]];
        const int32_t id1 = byte_to_id[p[1]];
        const uint64_t pair_key = (static_cast<uint64_t>(static_cast<uint32_t>(id0)) << 32) |
                                   static_cast<uint64_t>(static_cast<uint32_t>(id1));
        if (const MergeHashSlot* entry = lookup_merge(pair_key)) {
            output_ids[out_len++] = entry->merged_id;
        } else {
            output_ids[out_len++] = id0;
            if (out_len < output_cap) output_ids[out_len++] = id1;
        }
        return;
    }
    if (piece.size() > ENCODE_PIECE_MAX_BYTES) {
        size_t off = 0;
        while (off < piece.size() && out_len < output_cap) {
            size_t end = std::min(piece.size(), off + static_cast<size_t>(ENCODE_PIECE_MAX_BYTES));
            if (end < piece.size()) {
                for (size_t back = 0; back < 4 && end > off + 1 && (p[end] & 0xC0) == 0x80; ++back) --end;
            }
            encode_bpe_piece(piece.substr(off, end - off), scratch, output_ids, output_cap, out_len);
            off = end;
        }
        return;
    }

    PieceCacheEntry* cache_entry = nullptr;
    if (scratch.piece_cache_enabled && piece.size() <= 28) {
        cache_entry = &scratch.piece_cache[fast_piece_hash(piece) & (PieceCacheEntry::kCapacity - 1)];
        if (cache_entry->generation == scratch.piece_cache_generation &&
            cache_entry->byte_count == piece.size() &&
            std::memcmp(cache_entry->bytes, p, piece.size()) == 0) {
            const size_t take = std::min(static_cast<size_t>(cache_entry->token_count), output_cap - out_len);
            std::memcpy(output_ids + out_len, cache_entry->tokens, take * sizeof(int32_t));
            out_len += take;
            return;
        }
    }

    int32_t* ids = scratch.bpe_ids;
    size_t ids_len = piece.size();
    for (size_t idx = 0; idx < ids_len; ++idx) ids[idx] = byte_to_id[p[idx]];

    for (size_t step = 0; ids_len >= 2 && step < piece.size(); ++step) {
        uint32_t best_rank = UINT32_MAX;
        int32_t best_merged = -1;
        size_t best_idx = 0;

        for (size_t i = 0; i + 1 < ids_len; ++i) {
            const uint64_t pair_key = (static_cast<uint64_t>(static_cast<uint32_t>(ids[i])) << 32) |
                                       static_cast<uint64_t>(static_cast<uint32_t>(ids[i + 1]));
            if (const MergeHashSlot* entry = lookup_merge(pair_key)) {
                if (entry->rank < best_rank) {
                    best_rank = entry->rank;
                    best_merged = entry->merged_id;
                    best_idx = i;
                }
            }
        }
        if (best_rank == UINT32_MAX) break;

        const int32_t target_a = ids[best_idx];
        const int32_t target_b = ids[best_idx + 1];
        size_t write_idx = best_idx;
        size_t read_idx = best_idx;
        ids[write_idx++] = best_merged;
        read_idx += 2;
        while (read_idx < ids_len) {
            if (read_idx + 1 < ids_len && ids[read_idx] == target_a && ids[read_idx + 1] == target_b) {
                ids[write_idx++] = best_merged;
                read_idx += 2;
            } else {
                ids[write_idx++] = ids[read_idx++];
            }
        }
        SQUID_ASSERT(write_idx < ids_len);
        ids_len = write_idx;
    }

    if (cache_entry && ids_len <= 8) {
        std::memcpy(cache_entry->bytes, p, piece.size());
        std::memcpy(cache_entry->tokens, ids, ids_len * sizeof(int32_t));
        cache_entry->byte_count = static_cast<uint8_t>(piece.size());
        cache_entry->token_count = static_cast<uint8_t>(ids_len);
        cache_entry->generation = scratch.piece_cache_generation;
    }

    const size_t take = std::min(ids_len, output_cap - out_len);
    if (take > 0) {
        std::memcpy(output_ids + out_len, ids, take * sizeof(int32_t));
        out_len += take;
    }
}

void Tokenizer::pretokenize_and_encode(
    std::string_view text,
    TokenizerScratch& scratch,
    int32_t* output_ids,
    size_t output_cap,
    size_t& out_len) const {
    if (text.empty() || out_len >= output_cap) return;
    constexpr size_t kChunkCap = SCRATCH_CODEPOINTS_MAX;
    if (text.size() <= kChunkCap) {
        pretokenize_and_encode_chunk(text, scratch, output_ids, output_cap, out_len);
        return;
    }

    const auto* p = reinterpret_cast<const uint8_t*>(text.data());
    size_t off = 0;
    while (off < text.size() && out_len < output_cap) {
        size_t end = std::min(text.size(), off + kChunkCap);
        if (end < text.size()) {
            const size_t search_floor = (end > off + 65536) ? (end - 65536) : (off + 1);
            size_t split_pos = 0;
            for (size_t scan = end; scan > search_floor; ) {
                --scan;
                if (p[scan] == '\n') { split_pos = scan + 1; break; }
            }
            if (split_pos == 0) {
                for (size_t scan = end; scan > search_floor; ) {
                    --scan;
                    if (p[scan] == ' ' || p[scan] == '\t' || p[scan] == '\r') { split_pos = scan; break; }
                }
            }
            if (split_pos > off) {
                end = split_pos;
            } else {
                for (size_t back = 0; back < 4 && end > off + 1 && (p[end] & 0xC0) == 0x80; ++back) --end;
            }
        }
        pretokenize_and_encode_chunk(text.substr(off, end - off), scratch, output_ids, output_cap, out_len);
        off = end;
    }
}

void Tokenizer::pretokenize_and_encode_chunk(
    std::string_view text,
    TokenizerScratch& scratch,
    int32_t* output_ids,
    size_t output_cap,
    size_t& out_len) const {
    if (text.empty() || out_len >= output_cap) return;
    SQUID_ASSERT(text.size() <= SCRATCH_CODEPOINTS_MAX);

    uint32_t* cps = scratch.codepoints;
    uint32_t* offs = scratch.byte_offsets;
    size_t cp_count = 0;
    size_t bp = 0;

#if defined(__aarch64__)
    const auto* p = reinterpret_cast<const uint8_t*>(text.data());
    static constexpr uint32_t kLane0[4] = {0, 1, 2, 3};
    static constexpr uint32_t kLane1[4] = {4, 5, 6, 7};
    static constexpr uint32_t kLane2[4] = {8, 9, 10, 11};
    static constexpr uint32_t kLane3[4] = {12, 13, 14, 15};
    const uint32x4_t o0 = vld1q_u32(kLane0);
    const uint32x4_t o1 = vld1q_u32(kLane1);
    const uint32x4_t o2 = vld1q_u32(kLane2);
    const uint32x4_t o3 = vld1q_u32(kLane3);

    while (text.size() - bp >= 16) {
        uint8x16_t bytes = vld1q_u8(p + bp);
        if (vmaxvq_u8(bytes) < 0x80) {
            uint16x8_t lo16 = vmovl_u8(vget_low_u8(bytes));
            uint16x8_t hi16 = vmovl_u8(vget_high_u8(bytes));
            vst1q_u32(cps + cp_count + 0,  vmovl_u16(vget_low_u16(lo16)));
            vst1q_u32(cps + cp_count + 4,  vmovl_u16(vget_high_u16(lo16)));
            vst1q_u32(cps + cp_count + 8,  vmovl_u16(vget_low_u16(hi16)));
            vst1q_u32(cps + cp_count + 12, vmovl_u16(vget_high_u16(hi16)));

            uint32x4_t base = vdupq_n_u32(static_cast<uint32_t>(bp));
            vst1q_u32(offs + cp_count + 0,  vaddq_u32(base, o0));
            vst1q_u32(offs + cp_count + 4,  vaddq_u32(base, o1));
            vst1q_u32(offs + cp_count + 8,  vaddq_u32(base, o2));
            vst1q_u32(offs + cp_count + 12, vaddq_u32(base, o3));
            cp_count += 16;
            bp += 16;
        } else {
            const size_t block_end = bp + 16;
            while (bp < block_end) {
                offs[cp_count] = static_cast<uint32_t>(bp);
                cps[cp_count] = next_utf8(text, bp);
                ++cp_count;
            }
        }
    }
#endif
    while (bp < text.size()) {
        offs[cp_count] = static_cast<uint32_t>(bp);
        cps[cp_count] = next_utf8(text, bp);
        ++cp_count;
    }
    offs[cp_count] = static_cast<uint32_t>(text.size());

    size_t i = 0;
    while (i < cp_count && out_len < output_cap) {
        size_t end = match_letter_piece(cps, cp_count, i);
        if (end == 0) {
            if (cp_is_digit(cps[i])) {
                end = i + 1;
                if (end < cp_count && cp_is_digit(cps[end])) ++end;
                if (end < cp_count && cp_is_digit(cps[end])) ++end;
            } else {
                size_t j = i;
                if (cps[j] == ' ' && j + 1 < cp_count &&
                    !cp_is_space(cps[j + 1]) && !cp_is_letter(cps[j + 1]) && !cp_is_digit(cps[j + 1])) {
                    ++j;
                }
                if (!cp_is_space(cps[j]) && !cp_is_letter(cps[j]) && !cp_is_digit(cps[j])) {
                    while (j < cp_count && !cp_is_space(cps[j]) && !cp_is_letter(cps[j]) && !cp_is_digit(cps[j])) ++j;
                    while (j < cp_count && (cps[j] == '\r' || cps[j] == '\n' || cps[j] == '/')) ++j;
                    end = j;
                } else if (cp_is_space(cps[i])) {
                    size_t ws_end = i;
                    size_t last_nl = SIZE_MAX;
                    while (ws_end < cp_count && cp_is_space(cps[ws_end])) {
                        if (cp_is_newline(cps[ws_end])) last_nl = ws_end;
                        ++ws_end;
                    }
                    if (last_nl != SIZE_MAX) end = last_nl + 1;
                    else if (ws_end == cp_count) end = ws_end;
                    else if (ws_end - i > 1) end = ws_end - 1;
                    else end = i + 1;
                } else {
                    end = i + 1;
                }
            }
        }
        SQUID_ASSERT(end > i && end <= cp_count);
        encode_bpe_piece(text.substr(offs[i], offs[end] - offs[i]), scratch, output_ids, output_cap, out_len);
        i = end;
    }
}

bool parse_json_string(std::string_view data, size_t& pos, char* out_buf, size_t out_cap, std::string_view* out_view) {
    skip_ws(data, pos);
    if (pos >= data.size() || data[pos] != '"') return false;
    ++pos;

    size_t out_len = 0;
    bool overflowed = false;
    while (pos < data.size()) {
        const size_t plain_len = json_plain_prefix<false>(data.substr(pos));
        if (!overflowed && plain_len > 0) {
            const size_t take = std::min(plain_len, out_cap - out_len);
            std::memcpy(out_buf + out_len, data.data() + pos, take);
            out_len += take;
            overflowed = (take != plain_len);
        }
        pos += plain_len;
        if (pos == data.size()) break;
        const char c = data[pos++];
        if (c == '"') {
            if (overflowed) return false;
            *out_view = std::string_view(out_buf, out_len);
            return true;
        }
        if (pos >= data.size()) return false;
        const char esc = data[pos++];
        if (overflowed) {
            if (esc == 'u' && pos + 4 <= data.size()) pos += 4;
            continue;
        }
        if (esc == 'u') {
            uint32_t cp = 0;
            if (!parse_hex4(data, pos, &cp)) return false;
            pos += 4;
            if (cp >= 0xD800 && cp <= 0xDBFF && pos + 6 <= data.size() && data[pos] == '\\' && data[pos + 1] == 'u') {
                uint32_t lo = 0;
                if (!parse_hex4(data, pos + 2, &lo)) return false;
                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
                    pos += 6;
                }
            }
            if (!append_utf8_codepoint(out_buf, out_cap, out_len, cp)) overflowed = true;
        } else if (out_len == out_cap) {
            overflowed = true;
        } else {
            char ch = esc;
            switch (esc) {
                case 'b': ch = 0x08; break;
                case 'f': ch = 0x0c; break;
                case 'n': ch = '\n'; break;
                case 'r': ch = '\r'; break;
                case 't': ch = '\t'; break;
                default: break;
            }
            out_buf[out_len++] = ch;
        }
    }
    return false;
}

int64_t parse_i64(std::string_view data, size_t& pos) {
    skip_ws(data, pos);
    bool neg = false;
    if (pos < data.size() && data[pos] == '-') {
        neg = true;
        ++pos;
    }
    int64_t val = 0;
    for (uint32_t digits = 0; pos < data.size() && digits < 20; ++digits) {
        const char c = data[pos];
        if (c < '0' || c > '9') break;
        val = val * 10 + static_cast<int64_t>(c - '0');
        ++pos;
    }
    return neg ? -val : val;
}

void skip_json_value(std::string_view data, size_t& pos) {
    skip_ws(data, pos);
    if (pos >= data.size()) return;
    const char c0 = data[pos];
    if (c0 == '"') {
        skip_json_string(data, pos);
        return;
    }
    if (c0 == '{' || c0 == '[') {
        int depth = 0;
        while (pos < data.size()) {
            const char c = data[pos];
            if (c == '"') {
                skip_json_string(data, pos);
                continue;
            }
            if (c == '{' || c == '[') {
                ++depth;
            } else if (c == '}' || c == ']') {
                --depth;
                ++pos;
                if (depth == 0) return;
                continue;
            }
            ++pos;
        }
        return;
    }
    while (pos < data.size()) {
        const char c = data[pos];
        if (c == ',' || c == '}' || c == ']' || c == ' ' || c == '\t' || c == '\n' || c == '\r') break;
        ++pos;
    }
}

} // namespace squid
