#include "server.hpp"
#include "json_constraint.hpp"

#include "tiger.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <condition_variable>
#include <thread>
#include <vector>

namespace squid {

namespace {

constexpr size_t kRequestBufferBytes = 32 * 1024 * 1024;
constexpr size_t kPromptRenderBytes = 32 * 1024 * 1024;
constexpr size_t kResponseBufferBytes = 16 * 1024 * 1024;
constexpr size_t kContentBufferBytes = 8 * 1024 * 1024;
constexpr size_t kReasoningBufferBytes = 8 * 1024 * 1024;
constexpr size_t kToolRawBytes = 65536;
constexpr size_t kToolStoreBytes = 131072;
constexpr size_t kScratchStringBytes = 32 * 1024 * 1024;
constexpr size_t kSseFrameBytes = 64 * 1024;
constexpr size_t kPromptTokensMax = 1048576;
constexpr size_t kStopSequencesMax = 8;
constexpr size_t kStopSequenceMaxBytes = 64;
constexpr size_t kToolCallsMax = 16;
constexpr size_t kMaxImagesPerRequest = 16;

std::atomic<bool> g_stop_requested{false};
std::atomic<int> g_listen_fd{-1};
std::atomic<int> g_semif_fd{-1};

void handle_signal(int) {
    g_stop_requested.store(true, std::memory_order_release);
    int fd = g_listen_fd.exchange(-1, std::memory_order_acq_rel);
    if (fd >= 0) ::close(fd);
    int sfd = g_semif_fd.exchange(-1, std::memory_order_acq_rel);
    if (sfd >= 0) ::close(sfd);
}

void fill_random_hex(uint64_t& state, char* out, size_t len) {
    if (state == 0) state = 0x9e3779b97f4a7c15ULL;
    static constexpr char kHex[] = "0123456789abcdef";
    for (size_t i = 0; i < len; ++i) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        const uint64_t v = state * 0x2545F4914F6CDD1DULL;
        out[i] = kHex[(v >> 60) & 0x0f];
    }
}

bool constant_time_eql(std::string_view a, std::string_view b) {
    uint8_t diff = (a.size() == b.size()) ? 0 : 1;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        diff |= static_cast<uint8_t>(a[i]) ^ static_cast<uint8_t>(b[i]);
    }
    return diff == 0;
}

std::string_view trim_ascii_ws(std::string_view s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
    return s.substr(a, b - a);
}

bool eql_ignore_case(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        unsigned char ca = static_cast<unsigned char>(a[i]);
        unsigned char cb = static_cast<unsigned char>(b[i]);
        if (ca >= 'A' && ca <= 'Z') ca += 32;
        if (cb >= 'A' && cb <= 'Z') cb += 32;
        if (ca != cb) return false;
    }
    return true;
}

bool contains_ignore_case(std::string_view haystack, std::string_view needle) {
    if (needle.empty()) return true;
    if (haystack.size() < needle.size()) return false;
    for (size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
        if (eql_ignore_case(haystack.substr(i, needle.size()), needle)) return true;
    }
    return false;
}

int32_t clamp_i64_to_i32(int64_t v, int32_t lo, int32_t hi) {
    if (v < static_cast<int64_t>(lo)) return lo;
    if (v > static_cast<int64_t>(hi)) return hi;
    return static_cast<int32_t>(v);
}

bool send_all(int fd, std::string_view data) {
    size_t sent = 0;
    const size_t max_iter = data.size() + 64;
    for (size_t iter = 0; sent < data.size() && iter < max_iter; ++iter) {
        ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return sent == data.size();
}

void send_json_response(int fd, uint16_t status_code, std::string_view body) {
    const char* status_text = "OK";
    switch (status_code) {
        case 200: status_text = "OK"; break;
        case 400: status_text = "Bad Request"; break;
        case 401: status_text = "Unauthorized"; break;
        case 404: status_text = "Not Found"; break;
        case 413: status_text = "Payload Too Large"; break;
        case 500: status_text = "Internal Server Error"; break;
        case 501: status_text = "Not Implemented"; break;
        default: break;
    }
    char hdr[512];
    int n = std::snprintf(
        hdr, sizeof(hdr),
        "HTTP/1.1 %u %s\r\n"
        "Content-Type: application/json; charset=utf-8\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Content-Length: %zu\r\nConnection: close\r\n\r\n",
        status_code, status_text, body.size());
    if (n <= 0 || static_cast<size_t>(n) >= sizeof(hdr)) return;
    (void)send_all(fd, std::string_view(hdr, static_cast<size_t>(n)));
    (void)send_all(fd, body);
}

void send_error_json(int fd, uint16_t status_code, std::string_view message, std::string_view err_type, char* buf, size_t buf_cap) {
    ByteWriter bw(buf, buf_cap);
    bw.append_slice("{\"error\":{\"message\":");
    bw.append_json_escaped(message);
    bw.append_slice(",\"type\":");
    bw.append_json_escaped(err_type);
    bw.append_slice("}}");
    send_json_response(fd, status_code, bw.slice());
}

bool send_sse_headers(int fd) {
    return send_all(fd,
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Connection: close\r\n\r\n");
}

bool parse_json_i64_opt(std::string_view data, size_t& pos, int64_t* out) {
    skip_ws(data, pos);
    if (pos + 4 <= data.size() && data.substr(pos, 4) == "null") {
        pos += 4;
        return false;
    }
    const size_t start = pos;
    skip_json_value(data, pos);
    const std::string_view raw = trim_ascii_ws(data.substr(start, pos - start));
    if (raw.empty() || raw.size() >= 64) return false;
    char tmp[64];
    std::memcpy(tmp, raw.data(), raw.size());
    tmp[raw.size()] = '\0';
    char* endp = nullptr;
    long long iv = std::strtoll(tmp, &endp, 10);
    if (endp == tmp + raw.size()) {
        *out = static_cast<int64_t>(iv);
        return true;
    }
    endp = nullptr;
    double dv = std::strtod(tmp, &endp);
    if (endp == tmp + raw.size() && std::isfinite(dv)) {
        if (dv >= 9.0e18) *out = INT64_MAX;
        else if (dv <= -9.0e18) *out = INT64_MIN;
        else *out = static_cast<int64_t>(dv);
        return true;
    }
    return false;
}

bool parse_json_f64_opt(std::string_view data, size_t& pos, double* out) {
    skip_ws(data, pos);
    if (pos + 4 <= data.size() && data.substr(pos, 4) == "null") {
        pos += 4;
        return false;
    }
    const size_t start = pos;
    skip_json_value(data, pos);
    const std::string_view raw = trim_ascii_ws(data.substr(start, pos - start));
    if (raw.empty() || raw.size() >= 64) return false;
    char tmp[64];
    std::memcpy(tmp, raw.data(), raw.size());
    tmp[raw.size()] = '\0';
    char* endp = nullptr;
    double dv = std::strtod(tmp, &endp);
    if (endp != tmp + raw.size() || !std::isfinite(dv)) return false;
    *out = dv;
    return true;
}

bool parse_json_bool_opt(std::string_view data, size_t& pos, bool* out) {
    skip_ws(data, pos);
    if (pos + 4 <= data.size() && data.substr(pos, 4) == "true") {
        pos += 4;
        *out = true;
        return true;
    }
    if (pos + 5 <= data.size() && data.substr(pos, 5) == "false") {
        pos += 5;
        *out = false;
        return true;
    }
    skip_json_value(data, pos);
    return false;
}

bool parse_json_bool(std::string_view data, size_t& pos) {
    bool v = false;
    return parse_json_bool_opt(data, pos, &v) ? v : false;
}

bool extract_raw_field(std::string_view obj, std::string_view target_key, std::string_view* out) {
    size_t pos = 0;
    skip_ws(obj, pos);
    if (pos >= obj.size() || obj[pos] != '{') return false;
    ++pos;
    char kbuf[128];
    for (uint32_t iter = 0; pos < obj.size() && iter < 64; ++iter) {
        skip_ws(obj, pos);
        if (pos >= obj.size() || obj[pos] == '}') break;
        std::string_view k;
        if (!parse_json_string(obj, pos, kbuf, sizeof(kbuf), &k)) break;
        skip_ws(obj, pos);
        if (pos >= obj.size() || obj[pos] != ':') break;
        ++pos;
        skip_ws(obj, pos);
        const size_t start = pos;
        skip_json_value(obj, pos);
        if (k == target_key) {
            *out = obj.substr(start, pos - start);
            return true;
        }
        skip_ws(obj, pos);
        if (pos < obj.size() && obj[pos] == ',') ++pos;
    }
    return false;
}

bool extract_string_field(std::string_view obj, std::string_view target_key, char* out_buf, size_t out_cap, std::string_view* out) {
    std::string_view raw;
    if (!extract_raw_field(obj, target_key, &raw)) return false;
    size_t p = 0;
    return parse_json_string(raw, p, out_buf, out_cap, out);
}

bool extract_bool_field(std::string_view obj, std::string_view target_key) {
    std::string_view raw;
    if (!extract_raw_field(obj, target_key, &raw)) return false;
    size_t p = 0;
    return parse_json_bool(raw, p);
}

struct StopList {
    size_t count = 0;
    size_t lengths[kStopSequencesMax]{};
    char storage[kStopSequencesMax][kStopSequenceMaxBytes]{};
    size_t max_len = 0;

    void add(std::string_view text) {
        if (text.empty() || count >= kStopSequencesMax || text.size() > kStopSequenceMaxBytes) return;
        std::memcpy(storage[count], text.data(), text.size());
        lengths[count] = text.size();
        if (text.size() > max_len) max_len = text.size();
        ++count;
    }

    bool find_first(std::string_view haystack, size_t* out_pos) const {
        bool found = false;
        size_t best = 0;
        for (size_t i = 0; i < count; ++i) {
            std::string_view needle(storage[i], lengths[i]);
            const size_t p = haystack.find(needle);
            if (p != std::string_view::npos && (!found || p < best)) {
                best = p;
                found = true;
            }
        }
        if (found) *out_pos = best;
        return found;
    }
};

void parse_stop_field(std::string_view data, size_t& pos, StopList& stops) {
    skip_ws(data, pos);
    char tmp[kStopSequenceMaxBytes];
    std::string_view s;
    if (pos < data.size() && data[pos] == '"') {
        if (parse_json_string(data, pos, tmp, sizeof(tmp), &s)) stops.add(s);
    } else if (pos < data.size() && data[pos] == '[') {
        ++pos;
        for (uint32_t iter = 0; pos < data.size() && iter < 4096; ++iter) {
            skip_ws(data, pos);
            if (pos >= data.size()) break;
            if (data[pos] == ']') { ++pos; break; }
            if (data[pos] == '"') {
                if (parse_json_string(data, pos, tmp, sizeof(tmp), &s)) stops.add(s);
            } else {
                skip_json_value(data, pos);
            }
            skip_ws(data, pos);
            if (pos < data.size() && data[pos] == ',') ++pos;
        }
    } else {
        skip_json_value(data, pos);
    }
}

struct ToolCallRecord {
    char id_hex[12]{};
    size_t name_offset = 0;
    size_t name_len = 0;
    size_t args_offset = 0;
    size_t args_len = 0;
};

struct ServerBuffers {
    char* render_buf = nullptr;
    char* response_buf = nullptr;
    char* content_buf = nullptr;
    char* reasoning_buf = nullptr;
    char* tool_raw_buf = nullptr;
    char* tool_store_buf = nullptr;
    char* scratch_str_buf = nullptr;
    char* sse_buf = nullptr;
    int32_t* prompt_tokens = nullptr;
    float* logits_buf = nullptr;
    int32_t* order_buf = nullptr;   // [VOCAB_UNPADDED] candidate ordering for constrained decoding
    int32_t* gen_tokens = nullptr;  // [kPromptTokensMax] tokens generated by the current request
    TokenizerScratch tok_scratch{};

    void init() {
        render_buf = alloc_host_zeroed<char>(kPromptRenderBytes);
        response_buf = alloc_host_zeroed<char>(kResponseBufferBytes);
        content_buf = alloc_host_zeroed<char>(kContentBufferBytes);
        reasoning_buf = alloc_host_zeroed<char>(kReasoningBufferBytes);
        tool_raw_buf = alloc_host_zeroed<char>(kToolRawBytes);
        tool_store_buf = alloc_host_zeroed<char>(kToolStoreBytes);
        scratch_str_buf = alloc_host_zeroed<char>(kScratchStringBytes);
        sse_buf = alloc_host_zeroed<char>(kSseFrameBytes);
        prompt_tokens = alloc_host_zeroed<int32_t>(kPromptTokensMax);
        logits_buf = alloc_host_zeroed<float>(VOCAB_UNPADDED);
        order_buf = alloc_host_zeroed<int32_t>(VOCAB_UNPADDED);
        gen_tokens = alloc_host_zeroed<int32_t>(kPromptTokensMax);
        tok_scratch.init();
    }

    void deinit() {
        free_host(render_buf);
        free_host(response_buf);
        free_host(content_buf);
        free_host(reasoning_buf);
        free_host(tool_raw_buf);
        free_host(tool_store_buf);
        free_host(scratch_str_buf);
        free_host(sse_buf);
        free_host(prompt_tokens);
        free_host(logits_buf);
        free_host(order_buf);
        free_host(gen_tokens);
        tok_scratch.deinit();
    }
};

enum class InklingSection : uint8_t { None, Header, Thinking, Text, Tool };

struct InklingStreamParser {
    const Tokenizer* tokenizer = nullptr;
    bool raw_thinking = false;
    InklingSection section = InklingSection::None;
    char pending_utf8[16]{};
    size_t pending_utf8_len = 0;
    ByteWriter content_writer;
    ByteWriter reasoning_writer;
    ByteWriter tool_raw_writer;
    ByteWriter tool_store_writer;
    ToolCallRecord tool_calls[kToolCallsMax]{};
    size_t tool_call_count = 0;
    uint64_t prng_state = 0;

    struct FeedDelta {
        std::string_view reasoning;
        std::string_view content;
    };

    void init(const Tokenizer* tok, bool raw, ServerBuffers& buf, uint64_t seed) {
        tokenizer = tok;
        raw_thinking = raw;
        section = InklingSection::None;
        pending_utf8_len = 0;
        content_writer = ByteWriter(buf.content_buf, kContentBufferBytes);
        reasoning_writer = ByteWriter(buf.reasoning_buf, kReasoningBufferBytes);
        tool_raw_writer = ByteWriter(buf.tool_raw_buf, kToolRawBytes);
        tool_store_writer = ByteWriter(buf.tool_store_buf, kToolStoreBytes);
        tool_call_count = 0;
        prng_state = seed;
    }

    FeedDelta feed(int32_t token_id) {
        if (token_id == tokenizer->id_eos) return {};
        if (raw_thinking) {
            return push_utf8_bytes(tokenizer->decode_token_bytes(token_id, false), false);
        }
        if (token_id == tokenizer->id_message_model) {
            section = InklingSection::Header;
            pending_utf8_len = 0;
            tool_raw_writer.reset();
            return {};
        }
        if (token_id == tokenizer->id_thinking) {
            section = InklingSection::Thinking;
            pending_utf8_len = 0;
            return {};
        }
        if (token_id == tokenizer->id_text) {
            section = InklingSection::Text;
            pending_utf8_len = 0;
            return {};
        }
        if (tokenizer->id_tool_json >= 0 && token_id == tokenizer->id_tool_json) {
            section = InklingSection::Tool;
            tool_raw_writer.reset();
            return {};
        }
        if (token_id == tokenizer->id_end_message) {
            if (section == InklingSection::Tool && tool_raw_writer.length > 0) {
                parse_and_record_tool_call();
                tool_raw_writer.reset();
            }
            section = InklingSection::None;
            pending_utf8_len = 0;
            return {};
        }
        if (section == InklingSection::Header) return {};
        if (section == InklingSection::Tool) {
            (void)tool_raw_writer.try_append_slice(tokenizer->decode_token_bytes(token_id, true));
            return {};
        }
        return push_utf8_bytes(tokenizer->decode_token_bytes(token_id, false), section == InklingSection::Thinking);
    }

    FeedDelta push_utf8_bytes(std::string_view bytes, bool is_reasoning) {
        if (bytes.empty()) return {};
        char combined[256];
        std::string_view input;
        if (pending_utf8_len == 0) {
            input = bytes.substr(0, std::min(bytes.size(), sizeof(combined)));
        } else {
            std::memcpy(combined, pending_utf8, pending_utf8_len);
            const size_t take = std::min(bytes.size(), sizeof(combined) - pending_utf8_len);
            std::memcpy(combined + pending_utf8_len, bytes.data(), take);
            input = std::string_view(combined, pending_utf8_len + take);
        }
        const size_t valid_len = valid_utf8_prefix_len(input);
        const size_t rem_len = input.size() - valid_len;
        if (rem_len <= sizeof(pending_utf8)) {
            if (rem_len > 0) std::memcpy(pending_utf8, input.data() + valid_len, rem_len);
            pending_utf8_len = rem_len;
        } else {
            pending_utf8_len = 0;
        }
        if (valid_len == 0) return {};

        const std::string_view ready = input.substr(0, valid_len);
        if (is_reasoning) {
            const size_t start = reasoning_writer.length;
            if (!reasoning_writer.try_append_slice(ready)) return {};
            return {std::string_view(reasoning_writer.buffer + start, reasoning_writer.length - start), {}};
        } else {
            const size_t start = content_writer.length;
            if (!content_writer.try_append_slice(ready)) return {};
            return {{}, std::string_view(content_writer.buffer + start, content_writer.length - start)};
        }
    }

    void parse_and_record_tool_call() {
        if (tool_call_count >= kToolCallsMax) return;
        const std::string_view raw = tool_raw_writer.slice();
        size_t pos = 0;
        skip_ws(raw, pos);
        if (pos >= raw.size() || raw[pos] != '{') return;
        ++pos;

        char key_buf[128], name_buf[256];
        std::string_view name_slice;
        std::string_view args_slice = "{}";
        for (uint32_t iter = 0; pos < raw.size() && iter < 32; ++iter) {
            skip_ws(raw, pos);
            if (pos < raw.size() && raw[pos] == '}') break;
            std::string_view k;
            if (!parse_json_string(raw, pos, key_buf, sizeof(key_buf), &k)) break;
            skip_ws(raw, pos);
            if (pos >= raw.size() || raw[pos] != ':') break;
            ++pos;
            skip_ws(raw, pos);

            if (k == "name") {
                if (!parse_json_string(raw, pos, name_buf, sizeof(name_buf), &name_slice)) name_slice = {};
            } else if (k == "args" || k == "arguments") {
                const size_t start = pos;
                skip_json_value(raw, pos);
                args_slice = raw.substr(start, pos - start);
            } else {
                skip_json_value(raw, pos);
            }
            skip_ws(raw, pos);
            if (pos < raw.size() && raw[pos] == ',') ++pos;
        }
        if (name_slice.empty()) return;

        const size_t name_off = tool_store_writer.length;
        if (!tool_store_writer.try_append_slice(name_slice)) return;
        const size_t args_off = tool_store_writer.length;
        if (!tool_store_writer.try_append_slice(args_slice)) return;

        ToolCallRecord& rec = tool_calls[tool_call_count++];
        rec.name_offset = name_off;
        rec.name_len = name_slice.size();
        rec.args_offset = args_off;
        rec.args_len = args_slice.size();
        fill_random_hex(prng_state, rec.id_hex, sizeof(rec.id_hex));
    }
};

enum class DeltaKind { Role, Content, Reasoning, Empty };

ByteWriter chat_sse_frame_prefix(char* sse_buf, std::string_view req_id, int64_t created, std::string_view req_model) {
    ByteWriter bw(sse_buf, kSseFrameBytes);
    bw.append_slice("data: {\"id\":");
    bw.append_json_escaped(req_id);
    bw.append_fmt(",\"object\":\"chat.completion.chunk\",\"created\":%lld,\"model\":", static_cast<long long>(created));
    bw.append_json_escaped(req_model);
    bw.append_slice(",\"choices\":[{\"index\":0,\"delta\":{");
    return bw;
}

bool send_chat_sse_delta(
    int fd, ByteWriter frame_prefix, DeltaKind kind, std::string_view text,
    const char* finish_reason, const GenerateStats* usage) {
    ByteWriter bw = frame_prefix;
    switch (kind) {
        case DeltaKind::Role:
            bw.append_slice("\"role\":"); bw.append_json_escaped(text); break;
        case DeltaKind::Content:
            bw.append_slice("\"content\":"); bw.append_json_escaped(text); break;
        case DeltaKind::Reasoning:
            bw.append_slice("\"reasoning_content\":"); bw.append_json_escaped(text); break;
        case DeltaKind::Empty:
            break;
    }
    bw.append_slice("},\"finish_reason\":");
    if (finish_reason) bw.append_json_escaped(finish_reason);
    else bw.append_slice("null");
    bw.append_slice("}]");
    if (usage) {
        bw.append_fmt(",\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u,\"prompt_tokens_details\":{\"cached_tokens\":%u}}",
                      usage->prompt_tokens, usage->generated_tokens,
                      usage->prompt_tokens + usage->generated_tokens, usage->cached_tokens);
    }
    bw.append_slice("}\n\n");
    return send_all(fd, bw.slice());
}

bool send_chat_sse_tool_calls(int fd, ByteWriter frame_prefix, const InklingStreamParser& parser) {
    constexpr size_t kArgChunkBytes = 4096;
    for (size_t ti = 0; ti < parser.tool_call_count; ++ti) {
        const ToolCallRecord& tc = parser.tool_calls[ti];
        const std::string_view tname(parser.tool_store_writer.buffer + tc.name_offset, tc.name_len);
        const std::string_view targs(parser.tool_store_writer.buffer + tc.args_offset, tc.args_len);

        size_t first_end = std::min(targs.size(), kArgChunkBytes);
        if (first_end < targs.size()) {
            for (size_t back = 0; back < 4 && first_end > 1 && (static_cast<uint8_t>(targs[first_end]) & 0xC0) == 0x80; ++back) --first_end;
        }
        ByteWriter bw = frame_prefix;
        bool ok = true;
        ok = ok && bw.try_append_fmt("\"tool_calls\":[{\"index\":%zu,\"id\":\"call_", ti);
        ok = ok && bw.try_append_slice(std::string_view(tc.id_hex, sizeof(tc.id_hex)));
        ok = ok && bw.try_append_slice("\",\"type\":\"function\",\"function\":{\"name\":");
        ok = ok && bw.try_append_json_escaped(tname);
        ok = ok && bw.try_append_slice(",\"arguments\":");
        ok = ok && bw.try_append_json_escaped(targs.substr(0, first_end));
        ok = ok && bw.try_append_slice("}}]},\"finish_reason\":null}]}\n\n");
        if (!ok || !send_all(fd, bw.slice())) return false;

        size_t off = first_end;
        while (off < targs.size()) {
            size_t end = std::min(targs.size(), off + kArgChunkBytes);
            if (end < targs.size()) {
                for (size_t back = 0; back < 4 && end > off + 1 && (static_cast<uint8_t>(targs[end]) & 0xC0) == 0x80; ++back) --end;
            }
            bw = frame_prefix;
            ok = true;
            ok = ok && bw.try_append_fmt("\"tool_calls\":[{\"index\":%zu,\"function\":{\"arguments\":", ti);
            ok = ok && bw.try_append_json_escaped(targs.substr(off, end - off));
            ok = ok && bw.try_append_slice("}}]},\"finish_reason\":null}]}\n\n");
            if (!ok || !send_all(fd, bw.slice())) return false;
            off = end;
        }
    }
    return true;
}

bool json_array_has_items(std::string_view arr) {
    if (arr.size() < 2 || arr[0] != '[') return false;
    size_t p = 1;
    skip_ws(arr, p);
    return p < arr.size() && arr[p] != ']';
}

void append_encoded_segment(
    const Tokenizer& tokenizer, std::string_view text, bool parse_special,
    ServerBuffers& buffers, size_t& out_len, bool& overflow) {
    if (text.empty() || overflow) return;
    if (out_len >= kPromptTokensMax) { overflow = true; return; }
    const size_t added = tokenizer.encode_into(text, parse_special, buffers.tok_scratch,
                                               buffers.prompt_tokens + out_len, kPromptTokensMax - out_len);
    out_len += added;
    if (out_len >= kPromptTokensMax) overflow = true;
}

bool resolve_tool_name_by_id(std::string_view prior_prefix, std::string_view target_id, char* out_buf, size_t out_cap, std::string_view* out) {
    if (prior_prefix.size() < 2 || target_id.empty()) return false;
    size_t pos = 1;
    for (uint32_t miter = 0; pos < prior_prefix.size() && miter < 1024; ++miter) {
        skip_ws(prior_prefix, pos);
        if (pos >= prior_prefix.size() || prior_prefix[pos] != '{') {
            if (pos < prior_prefix.size()) ++pos;
            continue;
        }
        const size_t obj_start = pos;
        skip_json_value(prior_prefix, pos);
        const std::string_view msg_obj = prior_prefix.substr(obj_start, pos - obj_start);
        std::string_view tc_arr;
        if (extract_raw_field(msg_obj, "tool_calls", &tc_arr) && tc_arr.size() > 2 && tc_arr[0] == '[') {
            size_t tcp = 1;
            for (uint32_t tc_iter = 0; tcp < tc_arr.size() && tc_iter < 32; ++tc_iter) {
                skip_ws(tc_arr, tcp);
                if (tcp >= tc_arr.size() || tc_arr[tcp] == ']') break;
                const size_t tc_start = tcp;
                skip_json_value(tc_arr, tcp);
                const std::string_view tc_obj = tc_arr.substr(tc_start, tcp - tc_start);
                char id_buf[128];
                std::string_view id_val;
                if (extract_string_field(tc_obj, "id", id_buf, sizeof(id_buf), &id_val) && id_val == target_id) {
                    std::string_view fn_obj;
                    if (extract_raw_field(tc_obj, "function", &fn_obj)) {
                        return extract_string_field(fn_obj, "name", out_buf, out_cap, out);
                    }
                }
                skip_ws(tc_arr, tcp);
                if (tcp < tc_arr.size() && tc_arr[tcp] == ',') ++tcp;
            }
        }
        skip_ws(prior_prefix, pos);
        if (pos < prior_prefix.size() && prior_prefix[pos] == ',') ++pos;
    }
    return false;
}

struct RequestImages {
    PreprocessedImage items[kMaxImagesPerRequest]{};
    size_t count = 0;

    ~RequestImages() {
        for (size_t i = 0; i < count; ++i) items[i].deinit();
        count = 0;
    }
};

struct ChatEncodeResult {
    size_t prompt_len = 0;
    uint32_t system_prefix_tokens = 0;
    bool overflow = false;
    bool image_error = false;
};

ChatEncodeResult encode_chat_template_zero_alloc(
    const Tokenizer& tokenizer,
    std::string_view messages_json,
    std::string_view tools_json,
    std::string_view effort,
    bool keep_history_thinking,
    bool enable_thinking,
    ServerBuffers& buffers,
    RequestImages& images) {
    size_t out_len = 0;
    bool overflow = false;
    bool image_error = false;
    char* scratch = buffers.scratch_str_buf;

    if (json_array_has_items(tools_json)) {
        append_encoded_segment(tokenizer, "<|message_system|>tool_declare<|content_xml|>", true, buffers, out_len, overflow);
        ByteWriter rw(buffers.render_buf, kPromptRenderBytes);
        bool ok = rw.try_append_byte('[');
        size_t tp = 1;
        uint32_t emitted_tools = 0;
        for (uint32_t t_iter = 0; tp < tools_json.size() && t_iter < 256 && ok; ++t_iter) {
            skip_ws(tools_json, tp);
            if (tp >= tools_json.size() || tools_json[tp] == ']') break;
            const size_t t_start = tp;
            skip_json_value(tools_json, tp);
            const std::string_view tool_obj = tools_json.substr(t_start, tp - t_start);
            if (tool_obj.size() >= 2 && tool_obj[0] == '{') {
                std::string_view fn_obj = tool_obj;
                (void)extract_raw_field(tool_obj, "function", &fn_obj);
                char type_buf[64], tname_buf[256];
                std::string_view ttype = "function", tname = "", tdesc = "", tparams = "{}";
                (void)extract_string_field(tool_obj, "type", type_buf, sizeof(type_buf), &ttype);
                (void)extract_string_field(fn_obj, "name", tname_buf, sizeof(tname_buf), &tname);
                (void)extract_string_field(fn_obj, "description", scratch, kScratchStringBytes, &tdesc);
                (void)extract_raw_field(fn_obj, "parameters", &tparams);

                if (emitted_tools > 0) ok = ok && rw.try_append_byte(',');
                ok = ok && rw.try_append_slice("{\"description\":");
                ok = ok && rw.try_append_json_escaped(tdesc);
                ok = ok && rw.try_append_slice(",\"name\":");
                ok = ok && rw.try_append_json_escaped(tname);
                ok = ok && rw.try_append_slice(",\"parameters\":");
                ok = ok && rw.try_append_slice(tparams);
                ok = ok && rw.try_append_slice(",\"type\":");
                ok = ok && rw.try_append_json_escaped(ttype);
                ok = ok && rw.try_append_byte('}');
                ++emitted_tools;
            }
            skip_ws(tools_json, tp);
            if (tp < tools_json.size() && tools_json[tp] == ',') ++tp;
        }
        ok = ok && rw.try_append_byte(']');
        if (!ok) overflow = true;
        else append_encoded_segment(tokenizer, rw.slice(), false, buffers, out_len, overflow);
        append_encoded_segment(tokenizer, "<|end_message|>", true, buffers, out_len, overflow);
    }

    uint32_t system_prefix_tokens = static_cast<uint32_t>(std::min<size_t>(out_len, UINT32_MAX));
    bool effort_emitted = false;
    char effort_render[128];
    size_t pos = 1;
    for (uint32_t msg_iter = 0; pos < messages_json.size() && msg_iter < 1024 && !overflow; ++msg_iter) {
        skip_ws(messages_json, pos);
        if (pos >= messages_json.size() || messages_json[pos] == ']') break;
        if (messages_json[pos] != '{') {
            skip_json_value(messages_json, pos);
            skip_ws(messages_json, pos);
            if (pos < messages_json.size() && messages_json[pos] == ',') ++pos;
            continue;
        }
        const size_t obj_start = pos;
        skip_json_value(messages_json, pos);
        const std::string_view msg_obj = messages_json.substr(obj_start, pos - obj_start);

        char role_buf[32];
        std::string_view role = "user";
        (void)extract_string_field(msg_obj, "role", role_buf, sizeof(role_buf), &role);
        const bool is_sys = (role == "system" || role == "developer");
        if (!effort_emitted && !is_sys) {
            ByteWriter ew(effort_render, sizeof(effort_render));
            (void)Tokenizer::try_emit_thinking_effort(ew, effort);
            append_encoded_segment(tokenizer, ew.slice(), true, buffers, out_len, overflow);
            effort_emitted = true;
            system_prefix_tokens = static_cast<uint32_t>(std::min<size_t>(out_len, UINT32_MAX));
        }

        const std::string_view role_tok = (role == "user")      ? "<|message_user|>"
                                        : (role == "assistant") ? "<|message_model|>"
                                        : is_sys                ? "<|message_system|>"
                                        : (role == "tool")      ? "<|message_tool|>"
                                                                : "<|message_user|>";

        char name_buf[128];
        std::string_view resolved_name;
        bool has_resolved_name = extract_string_field(msg_obj, "name", name_buf, sizeof(name_buf), &resolved_name);
        if (role == "tool" && (!has_resolved_name || resolved_name.empty())) {
            char tcid_buf[128];
            std::string_view tcid;
            if (extract_string_field(msg_obj, "tool_call_id", tcid_buf, sizeof(tcid_buf), &tcid)) {
                has_resolved_name = resolve_tool_name_by_id(messages_json.substr(0, obj_start), tcid, name_buf, sizeof(name_buf), &resolved_name);
            }
        }

        const bool is_assistant = (role == "assistant");
        bool emitted_assistant_block = false;
        bool has_reasoning = false;
        if (is_assistant && keep_history_thinking) {
            std::string_view rc;
            if (extract_string_field(msg_obj, "reasoning_content", scratch, kScratchStringBytes, &rc) && !rc.empty()) {
                has_reasoning = true;
                emitted_assistant_block = true;
                append_encoded_segment(tokenizer, role_tok, true, buffers, out_len, overflow);
                if (has_resolved_name) append_encoded_segment(tokenizer, resolved_name, false, buffers, out_len, overflow);
                append_encoded_segment(tokenizer, "<|content_thinking|>", true, buffers, out_len, overflow);
                append_encoded_segment(tokenizer, rc, false, buffers, out_len, overflow);
                append_encoded_segment(tokenizer, "<|end_message|>", true, buffers, out_len, overflow);
            }
        }

        std::string_view tc_arr;
        const bool has_tool_calls = is_assistant && extract_raw_field(msg_obj, "tool_calls", &tc_arr) && json_array_has_items(tc_arr);

        std::string_view c_raw;
        if (extract_raw_field(msg_obj, "content", &c_raw)) {
            if (!c_raw.empty() && c_raw[0] == '"') {
                size_t cp = 0;
                std::string_view cs;
                if (parse_json_string(c_raw, cp, scratch, kScratchStringBytes, &cs)) {
                    if (!cs.empty() || !(is_assistant && (has_tool_calls || has_reasoning))) {
                        emitted_assistant_block = true;
                        append_encoded_segment(tokenizer, role_tok, true, buffers, out_len, overflow);
                        if (has_resolved_name) append_encoded_segment(tokenizer, resolved_name, false, buffers, out_len, overflow);
                        append_encoded_segment(tokenizer, "<|content_text|>", true, buffers, out_len, overflow);
                        append_encoded_segment(tokenizer, cs, false, buffers, out_len, overflow);
                        append_encoded_segment(tokenizer, "<|end_message|>", true, buffers, out_len, overflow);
                    }
                }
            } else if (!c_raw.empty() && c_raw[0] == '[') {
                size_t ap = 1;
                for (uint32_t piter = 0; ap < c_raw.size() && piter < 64 && !overflow; ++piter) {
                    skip_ws(c_raw, ap);
                    if (ap >= c_raw.size() || c_raw[ap] == ']') break;
                    if (c_raw[ap] == '"') {
                        std::string_view ps;
                        if (parse_json_string(c_raw, ap, scratch, kScratchStringBytes, &ps)) {
                            emitted_assistant_block = true;
                            append_encoded_segment(tokenizer, role_tok, true, buffers, out_len, overflow);
                            if (has_resolved_name) append_encoded_segment(tokenizer, resolved_name, false, buffers, out_len, overflow);
                            append_encoded_segment(tokenizer, "<|content_text|>", true, buffers, out_len, overflow);
                            append_encoded_segment(tokenizer, ps, false, buffers, out_len, overflow);
                            append_encoded_segment(tokenizer, "<|end_message|>", true, buffers, out_len, overflow);
                        }
                    } else if (c_raw[ap] == '{') {
                        const size_t ps_start = ap;
                        skip_json_value(c_raw, ap);
                        const std::string_view part_obj = c_raw.substr(ps_start, ap - ps_start);
                        char ptype_buf[32];
                        std::string_view ptype;
                        const bool has_ptype = extract_string_field(part_obj, "type", ptype_buf, sizeof(ptype_buf), &ptype);
                        if (has_ptype && (ptype == "image_url" || ptype == "image" || ptype == "input_image")) {
                            std::string_view img_src;
                            std::string_view iu_raw;
                            if (extract_raw_field(part_obj, "image_url", &iu_raw)) {
                                if (!iu_raw.empty() && iu_raw[0] == '{') {
                                    (void)extract_string_field(iu_raw, "url", scratch, kScratchStringBytes, &img_src);
                                } else if (!iu_raw.empty() && iu_raw[0] == '"') {
                                    size_t ip = 0;
                                    (void)parse_json_string(iu_raw, ip, scratch, kScratchStringBytes, &img_src);
                                }
                            }
                            if (img_src.empty()) {
                                (void)extract_string_field(part_obj, "image", scratch, kScratchStringBytes, &img_src);
                            }
                            if (img_src.empty()) {
                                (void)extract_string_field(part_obj, "url", scratch, kScratchStringBytes, &img_src);
                            }
                            if (img_src.empty() || images.count >= kMaxImagesPerRequest ||
                                !preprocess_image_source(img_src, images.items[images.count])) {
                                image_error = true;
                                break;
                            }
                            const int n_img_toks = images.items[images.count].num_tokens;
                            ++images.count;
                            emitted_assistant_block = true;
                            append_encoded_segment(tokenizer, role_tok, true, buffers, out_len, overflow);
                            if (has_resolved_name) append_encoded_segment(tokenizer, resolved_name, false, buffers, out_len, overflow);
                            append_encoded_segment(tokenizer, "<|content_image|>", true, buffers, out_len, overflow);
                            if (out_len + static_cast<size_t>(n_img_toks) > kPromptTokensMax) {
                                overflow = true;
                            } else {
                                for (int it = 0; it < n_img_toks; ++it) {
                                    buffers.prompt_tokens[out_len++] = IMAGE_TOKEN_ID;
                                }
                            }
                            append_encoded_segment(tokenizer, "<|end_message|>", true, buffers, out_len, overflow);
                        } else {
                            std::string_view ts;
                            if (extract_string_field(part_obj, "text", scratch, kScratchStringBytes, &ts)) {
                                emitted_assistant_block = true;
                                append_encoded_segment(tokenizer, role_tok, true, buffers, out_len, overflow);
                                if (has_resolved_name) append_encoded_segment(tokenizer, resolved_name, false, buffers, out_len, overflow);
                                append_encoded_segment(tokenizer, "<|content_text|>", true, buffers, out_len, overflow);
                                append_encoded_segment(tokenizer, ts, false, buffers, out_len, overflow);
                                append_encoded_segment(tokenizer, "<|end_message|>", true, buffers, out_len, overflow);
                            }
                        }
                    } else {
                        skip_json_value(c_raw, ap);
                    }
                    skip_ws(c_raw, ap);
                    if (ap < c_raw.size() && c_raw[ap] == ',') ++ap;
                }
            }
        }

        if (has_tool_calls) {
            size_t tcp = 1;
            for (uint32_t tc_iter = 0; tcp < tc_arr.size() && tc_iter < 32 && !overflow; ++tc_iter) {
                skip_ws(tc_arr, tcp);
                if (tcp >= tc_arr.size() || tc_arr[tcp] == ']') break;
                const size_t tc_start = tcp;
                skip_json_value(tc_arr, tcp);
                const std::string_view tc_obj = tc_arr.substr(tc_start, tcp - tc_start);
                std::string_view fn_obj;
                if (extract_raw_field(tc_obj, "function", &fn_obj)) {
                    char fname_buf[256];
                    std::string_view fname;
                    if (extract_string_field(fn_obj, "name", fname_buf, sizeof(fname_buf), &fname)) {
                        emitted_assistant_block = true;
                        append_encoded_segment(tokenizer, "<|message_model|>", true, buffers, out_len, overflow);
                        append_encoded_segment(tokenizer, fname, false, buffers, out_len, overflow);
                        append_encoded_segment(tokenizer, "<|content_invoke_tool_json|>", true, buffers, out_len, overflow);
                        ByteWriter tw(buffers.render_buf, kPromptRenderBytes);
                        bool tok_ok = tw.try_append_slice("{\"name\":");
                        tok_ok = tok_ok && tw.try_append_json_escaped(fname);
                        tok_ok = tok_ok && tw.try_append_slice(",\"args\":");
                        std::string_view args_raw;
                        if (extract_raw_field(fn_obj, "arguments", &args_raw)) {
                            if (!args_raw.empty() && args_raw[0] == '"') {
                                size_t ap2 = 0;
                                std::string_view args_str;
                                if (parse_json_string(args_raw, ap2, scratch, kScratchStringBytes, &args_str)) {
                                    const std::string_view trimmed = trim_ascii_ws(args_str);
                                    tok_ok = tok_ok && tw.try_append_slice(trimmed.empty() ? "{}" : trimmed);
                                } else {
                                    tok_ok = tok_ok && tw.try_append_slice("{}");
                                }
                            } else {
                                tok_ok = tok_ok && tw.try_append_slice(args_raw);
                            }
                        } else {
                            tok_ok = tok_ok && tw.try_append_slice("{}");
                        }
                        tok_ok = tok_ok && tw.try_append_byte('}');
                        if (!tok_ok) overflow = true;
                        else append_encoded_segment(tokenizer, tw.slice(), false, buffers, out_len, overflow);
                        append_encoded_segment(tokenizer, "<|end_message|>", true, buffers, out_len, overflow);
                    }
                }
                skip_ws(tc_arr, tcp);
                if (tcp < tc_arr.size() && tc_arr[tcp] == ',') ++tcp;
            }
        }

        if (is_assistant && emitted_assistant_block) {
            append_encoded_segment(tokenizer, "<|content_model_end_sampling|>", true, buffers, out_len, overflow);
        }
        if (is_sys && !effort_emitted) {
            system_prefix_tokens = static_cast<uint32_t>(std::min<size_t>(out_len, UINT32_MAX));
        }

        skip_ws(messages_json, pos);
        if (pos < messages_json.size() && messages_json[pos] == ',') ++pos;
    }

    if (!effort_emitted && !overflow) {
        ByteWriter ew(effort_render, sizeof(effort_render));
        (void)Tokenizer::try_emit_thinking_effort(ew, effort);
        append_encoded_segment(tokenizer, ew.slice(), true, buffers, out_len, overflow);
        system_prefix_tokens = static_cast<uint32_t>(std::min<size_t>(out_len, UINT32_MAX));
    }
    if (!enable_thinking && !overflow) {
        append_encoded_segment(tokenizer, "<|message_model|><|content_text|>", true, buffers, out_len, overflow);
    }
    return {out_len, system_prefix_tokens, overflow, image_error};
}

struct ParsedHttp {
    std::string_view method;
    std::string_view path;
    std::string_view auth_header;
    std::string_view body;
};

enum class HttpReadStatus { Ok, BadRequest, PayloadTooLarge, NotImplemented, Closed };

struct HttpReadResult {
    HttpReadStatus status = HttpReadStatus::Closed;
    ParsedHttp req{};
    const char* err_msg = "";
};

HttpReadResult read_http_request(int fd, char* buffer, size_t buf_cap) {
    SQUID_ASSERT(fd >= 0 && buf_cap > 4096);
    size_t received = 0;
    size_t header_end = SIZE_MAX;

    for (uint32_t riter = 0; header_end == SIZE_MAX && riter < 4096; ++riter) {
        if (received >= 65536) return {HttpReadStatus::BadRequest, {}, "HTTP headers exceed 64 KiB limit"};
        const size_t want = std::min<size_t>(8192, buf_cap - received);
        ssize_t n = ::recv(fd, buffer + received, want, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return {HttpReadStatus::Closed, {}, ""};
        }
        const size_t scan_start = (received >= 3) ? (received - 3) : 0;
        received += static_cast<size_t>(n);
        const std::string_view seg(buffer + scan_start, received - scan_start);
        const size_t off = seg.find("\r\n\r\n");
        if (off != std::string_view::npos) header_end = scan_start + off;
    }
    if (header_end == SIZE_MAX) return {HttpReadStatus::Closed, {}, ""};

    const std::string_view hdr_block(buffer, header_end);
    const size_t line_end_pos = hdr_block.find("\r\n");
    const size_t line_end = (line_end_pos == std::string_view::npos) ? hdr_block.size() : line_end_pos;
    const std::string_view req_line = hdr_block.substr(0, line_end);
    const size_t sp1 = req_line.find(' ');
    if (sp1 == std::string_view::npos) return {HttpReadStatus::BadRequest, {}, "Malformed HTTP request line"};
    const size_t sp2 = req_line.find(' ', sp1 + 1);
    if (sp2 == std::string_view::npos) return {HttpReadStatus::BadRequest, {}, "Malformed HTTP request line"};

    const std::string_view method = req_line.substr(0, sp1);
    const std::string_view full_path = req_line.substr(sp1 + 1, sp2 - (sp1 + 1));
    const size_t q = full_path.find('?');
    const std::string_view path = (q == std::string_view::npos) ? full_path : full_path.substr(0, q);

    size_t content_len = 0;
    bool has_content_len = false;
    std::string_view auth_header;
    bool expect_continue = false;
    bool chunked_te = false;

    size_t pos = (line_end + 2 <= hdr_block.size()) ? (line_end + 2) : hdr_block.size();
    for (uint32_t hiter = 0; pos < hdr_block.size() && hiter < 128; ++hiter) {
        const size_t nr = hdr_block.find("\r\n", pos);
        const std::string_view line = (nr == std::string_view::npos) ? hdr_block.substr(pos) : hdr_block.substr(pos, nr - pos);
        const size_t colon = line.find(':');
        if (colon != std::string_view::npos) {
            const std::string_view hname = trim_ascii_ws(line.substr(0, colon));
            const std::string_view hval = trim_ascii_ws(line.substr(colon + 1));
            if (eql_ignore_case(hname, "content-length")) {
                if (hval.empty() || hval.size() >= 32) return {HttpReadStatus::BadRequest, {}, "Invalid Content-Length header"};
                char tmp[32];
                std::memcpy(tmp, hval.data(), hval.size());
                tmp[hval.size()] = '\0';
                char* endp = nullptr;
                unsigned long long parsed = std::strtoull(tmp, &endp, 10);
                if (endp != tmp + hval.size()) return {HttpReadStatus::BadRequest, {}, "Invalid Content-Length header"};
                if (has_content_len && static_cast<size_t>(parsed) != content_len) {
                    return {HttpReadStatus::BadRequest, {}, "Conflicting Content-Length headers"};
                }
                content_len = static_cast<size_t>(parsed);
                has_content_len = true;
            } else if (eql_ignore_case(hname, "transfer-encoding")) {
                if (contains_ignore_case(hval, "chunked")) chunked_te = true;
            } else if (eql_ignore_case(hname, "authorization")) {
                auth_header = hval;
            } else if (eql_ignore_case(hname, "expect") && eql_ignore_case(hval, "100-continue")) {
                expect_continue = true;
            }
        }
        if (nr == std::string_view::npos) break;
        pos = nr + 2;
    }

    if (chunked_te) return {HttpReadStatus::NotImplemented, {}, "Transfer-Encoding: chunked is not supported"};
    const size_t body_start = header_end + 4;
    if (content_len > buf_cap - body_start) return {HttpReadStatus::PayloadTooLarge, {}, ""};
    if (expect_continue) (void)send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n");

    for (uint32_t biter = 0; received < body_start + content_len && biter < 262144; ++biter) {
        ssize_t n = ::recv(fd, buffer + received, body_start + content_len - received, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return {HttpReadStatus::Closed, {}, ""};
        }
        received += static_cast<size_t>(n);
    }
    if (received < body_start + content_len) return {HttpReadStatus::Closed, {}, ""};

    return {HttpReadStatus::Ok, {method, path, auth_header, std::string_view(buffer + body_start, content_len)}, ""};
}

// Picks a token for which valid(t) holds, from one step's logits: greedy at temperature 0,
// otherwise sampled among the best valid candidates with temperature/top_p/top_k.
// `order` is [VOCAB_UNPADDED] scratch. Returns -1 if no token is valid.
template <class Valid>
int32_t pick_valid_token(const float* logits, int32_t* order, float temperature, float top_p, int32_t top_k,
                         uint64_t& rng, Valid&& valid) {
    const int32_t V = VOCAB_UNPADDED;
    const int32_t want = (temperature < 1e-4f) ? 1 : (top_k > 0 ? std::min(top_k, 64) : 64);
    constexpr int32_t kTop = 2048;
    auto by_logit = [logits](int32_t a, int32_t b) { return logits[a] > logits[b]; };
    std::iota(order, order + V, 0);
    std::nth_element(order, order + kTop, order + V, by_logit);
    std::sort(order, order + kTop, by_logit);
    int32_t picked[64];
    int32_t n = 0;
    for (int32_t i = 0; i < kTop && n < want; ++i) {
        if (valid(order[i])) picked[n++] = order[i];
    }
    if (n == 0) {
        // Rare: nothing the model favours is valid here. Scan the rest of the vocabulary.
        std::sort(order + kTop, order + V, by_logit);
        for (int32_t i = kTop; i < V && n < want; ++i) {
            if (valid(order[i])) picked[n++] = order[i];
        }
    }
    if (n == 0) return -1;
    if (n == 1) return picked[0];
    double w[64];
    double z = 0.0;
    const double mx = logits[picked[0]];
    for (int32_t i = 0; i < n; ++i) { w[i] = std::exp((logits[picked[i]] - mx) / temperature); z += w[i]; }
    int32_t keep = n;
    double cum = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        cum += w[i] / z;
        if (cum >= top_p) { keep = i + 1; break; }
    }
    double zk = 0.0;
    for (int32_t i = 0; i < keep; ++i) zk += w[i];
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    double r = static_cast<double>(rng >> 11) * (1.0 / 9007199254740992.0) * zk;
    for (int32_t i = 0; i < keep; ++i) {
        r -= w[i];
        if (r <= 0.0) return picked[i];
    }
    return picked[keep - 1];
}

// response_format: picks every generated token so the final text is JSON (json_object) or
// matches a JSON Schema (json_schema). Thinking, if enabled, is left free; only the text section
// is constrained, and tool calls are not allowed.
struct JsonConstraintCtx {
    enum Phase : uint8_t { Pre, Text, Post };
    const Tokenizer* tok = nullptr;
    JsonMatcher matcher;
    Phase phase = Pre;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int32_t top_k = 0;
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    int32_t* order = nullptr;
    uint32_t replaced = 0;

    void init(const Tokenizer* t, const JsonSchema* schema, bool thinking, const SampleParams& sp, int32_t* order_buf) {
        tok = t;
        matcher.init(schema);
        phase = thinking ? Pre : Text;
        temperature = sp.temperature;
        top_p = (sp.top_p > 0.0f && sp.top_p <= 1.0f) ? sp.top_p : 1.0f;
        top_k = sp.top_k;
        rng = sp.seed ? sp.seed : 0x9e3779b97f4a7c15ULL;
        order = order_buf;
        replaced = 0;
    }

    bool valid(int32_t t) const {
        switch (phase) {
            case Pre:
                return t != tok->id_eos && t != tok->id_tool_json;
            case Text: {
                if (t == tok->id_end_message || t == tok->id_eos) return matcher.can_end();
                const std::string_view b = tok->decode_token_bytes(t, true);
                return !b.empty() && matcher.accepts(b);
            }
            case Post:
                return t == tok->id_eos;
        }
        return false;
    }

    void accept(int32_t t) {
        if (phase == Pre) {
            if (t == tok->id_text) phase = Text;
            return;
        }
        if (phase == Text) {
            if (t == tok->id_end_message || t == tok->id_eos) { phase = Post; return; }
            (void)matcher.advance(tok->decode_token_bytes(t, true));
        }
    }

    int32_t choose(int32_t proposed, const float* logits) {
        if (valid(proposed)) { accept(proposed); return proposed; }
        if (phase == Post) { accept(tok->id_eos); return tok->id_eos; }
        if (logits == nullptr) return kConstraintNeedLogits;
        ++replaced;
        const int32_t choice = pick_valid_token(logits, order, temperature, top_p, top_k, rng,
                                                [this](int32_t t) { return valid(t); });
        if (choice < 0) return proposed;  // nothing valid at all; leave the model's choice
        accept(choice);
        return choice;
    }

    static int32_t choose_cb(void* ctx, int32_t proposed, const float* logits) {
        return static_cast<JsonConstraintCtx*>(ctx)->choose(proposed, logits);
    }
};

// Tool calls: when a request has tools (and no response_format), the arguments JSON after
// <|content_invoke_tool_json|> must be {"name": <a tool>, "args": <that tool's parameters>}.
// The model names the tool in the message header first, so the matcher is rooted at that
// tool's schema; if the header is not a tool name, the name must still be one of the tools.
// Everything outside the arguments JSON is left to the model.
struct ToolSchemas {
    JsonSchema schema;
    std::vector<std::string> names;
    std::vector<int32_t> entries;

    // tools_json is the request's "tools" array.
    bool build(std::string_view tools_json, char* scratch, size_t scratch_cap) {
        schema.init_empty();
        names.clear();
        entries.clear();
        size_t tp = 1;
        for (uint32_t it = 0; tp < tools_json.size() && it < 256; ++it) {
            skip_ws(tools_json, tp);
            if (tp >= tools_json.size() || tools_json[tp] == ']') break;
            const size_t t_start = tp;
            skip_json_value(tools_json, tp);
            const std::string_view tool_obj = tools_json.substr(t_start, tp - t_start);
            std::string_view fn_obj = tool_obj;
            (void)extract_raw_field(tool_obj, "function", &fn_obj);
            std::string_view tname, tparams;
            if (extract_string_field(fn_obj, "name", scratch, scratch_cap, &tname) && !tname.empty()) {
                const std::string name(tname);
                if (!extract_raw_field(fn_obj, "parameters", &tparams)) tparams = {};
                entries.push_back(schema.add_tool(name, trim_ascii_ws(tparams)));
                names.push_back(name);
            }
            skip_ws(tools_json, tp);
            if (tp < tools_json.size() && tools_json[tp] == ',') ++tp;
        }
        if (names.empty()) return false;
        schema.finish_tools(names);
        return true;
    }
};

struct ToolCallConstraintCtx {
    enum Phase : uint8_t { Free, Header, Args };
    const Tokenizer* tok = nullptr;
    const ToolSchemas* tools = nullptr;
    JsonMatcher matcher;
    Phase phase = Free;
    char header[256];
    size_t header_len = 0;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int32_t top_k = 0;
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    int32_t* order = nullptr;
    uint32_t replaced = 0;

    void init(const Tokenizer* t, const ToolSchemas* ts, const SampleParams& sp, int32_t* order_buf) {
        tok = t;
        tools = ts;
        phase = Free;
        header_len = 0;
        temperature = sp.temperature;
        top_p = (sp.top_p > 0.0f && sp.top_p <= 1.0f) ? sp.top_p : 1.0f;
        top_k = sp.top_k;
        rng = sp.seed ? sp.seed : 0x9e3779b97f4a7c15ULL;
        order = order_buf;
        replaced = 0;
    }

    bool valid(int32_t t) const {
        if (phase != Args) return true;
        if (t == tok->id_end_message) return matcher.can_end();
        const std::string_view b = tok->decode_token_bytes(t, true);
        return !b.empty() && matcher.accepts(b);
    }

    void accept(int32_t t) {
        if (phase == Args) {
            if (t == tok->id_end_message) phase = Free;
            else (void)matcher.advance(tok->decode_token_bytes(t, true));
            return;
        }
        if (t == tok->id_message_model) { phase = Header; header_len = 0; return; }
        if (tok->id_tool_json >= 0 && t == tok->id_tool_json) {
            const std::string_view h = trim_ascii_ws(std::string_view(header, header_len));
            int32_t root = tools->schema.root;
            for (size_t i = 0; i < tools->names.size(); ++i) {
                if (tools->names[i] == h) { root = tools->entries[i]; break; }
            }
            matcher.init(&tools->schema, root);
            phase = Args;
            return;
        }
        if (phase == Header) {
            const std::string_view b = tok->decode_token_bytes(t, true);
            if (b.empty()) { phase = Free; return; }  // e.g. <|content_text|>: an ordinary message
            const size_t n = std::min(b.size(), sizeof(header) - header_len);
            std::memcpy(header + header_len, b.data(), n);
            header_len += n;
        }
    }

    int32_t choose(int32_t proposed, const float* logits) {
        if (valid(proposed)) { accept(proposed); return proposed; }
        if (logits == nullptr) return kConstraintNeedLogits;
        ++replaced;
        const int32_t choice = pick_valid_token(logits, order, temperature, top_p, top_k, rng,
                                                [this](int32_t t) { return valid(t); });
        if (choice < 0) return proposed;
        accept(choice);
        return choice;
    }

    static int32_t choose_cb(void* ctx, int32_t proposed, const float* logits) {
        return static_cast<ToolCallConstraintCtx*>(ctx)->choose(proposed, logits);
    }
};

// Degenerate-repetition guard. Watches generated tokens (gen[], written by the engine before
// each callback) for an exact cycle: the last 2*P tokens repeating the P before them (three
// copies, at least kMinSpan tokens) for some period P <= kMaxPeriod. In reasoning, and when
// reasoning exceeds max_reasoning tokens, it asks the caller to end the reasoning block and let
// the model answer; anywhere else it stops generation.
struct LoopGuard {
    enum Act : uint8_t { None, EndThinking, Stop };
    static constexpr uint32_t kMaxPeriod = 256;
    // Reasoning: three copies spanning >= 48 tokens. Answers and tool calls can repeat
    // legitimately (tables, arrays), so there: four copies spanning >= 200 tokens.
    static constexpr uint32_t kMinSpanThinking = 48;
    static constexpr uint32_t kMinSpanText = 200;
    bool enabled = true;
    const int32_t* gen = nullptr;
    uint32_t base = 0;          // index in gen[] of the first token this guard has seen
    uint32_t n = 0;             // tokens seen since base
    uint32_t run[kMaxPeriod + 1];
    uint32_t reasoning_tokens = 0;
    uint32_t max_reasoning = 0; // 0 = no cap
    Act pending = None;
    uint32_t cut = 0;           // gen[] index to keep up to (exclusive) when acting
    uint32_t period = 0;
    bool stopped = false;

    void start(const int32_t* g, uint32_t at) {
        gen = g;
        base = at;
        n = 0;
        std::memset(run, 0, sizeof(run));
        pending = None;
    }

    Act observe(InklingSection sec) {
        if (!enabled) return None;
        const uint32_t i = base + n;  // index of the token just generated
        ++n;
        if (sec == InklingSection::Thinking) ++reasoning_tokens;
        const bool thinking = (sec == InklingSection::Thinking);
        const uint32_t copies = thinking ? 2 : 3;  // repeats after the first copy
        const uint32_t min_span = thinking ? kMinSpanThinking : kMinSpanText;
        const uint32_t pmax = std::min<uint32_t>(kMaxPeriod, n - 1);
        for (uint32_t p = 1; p <= pmax; ++p) {
            run[p] = (gen[i] == gen[i - p]) ? run[p] + 1 : 0;
            if (run[p] >= copies * p && run[p] + p >= min_span) {
                period = p;
                cut = i + 1 - run[p];  // keep the first copy, drop the repeats
                pending = (sec == InklingSection::Thinking) ? EndThinking : Stop;
                return pending;
            }
        }
        if (max_reasoning > 0 && sec == InklingSection::Thinking && reasoning_tokens >= max_reasoning) {
            period = 0;
            cut = i + 1;
            pending = EndThinking;
            return pending;
        }
        return None;
    }
};

// Returns the index of the <|content_invoke_tool_json|> token of the first tool call in
// gen[0, n) whose arguments do not match the tool's schema, or -1 if all are valid (a call cut
// off by max_tokens counts as valid: constraining it again would not finish it either).
int64_t find_invalid_tool_call(const Tokenizer& tok, const ToolSchemas& ts, const int32_t* gen, size_t n,
                               int32_t* order_buf) {
    ToolCallConstraintCtx scan{};
    scan.init(&tok, &ts, SampleParams{}, order_buf);
    int64_t call_at = -1;
    for (size_t i = 0; i < n; ++i) {
        const int32_t t = gen[i];
        if (scan.phase == ToolCallConstraintCtx::Args && !scan.valid(t)) return call_at;
        if (tok.id_tool_json >= 0 && t == tok.id_tool_json) call_at = static_cast<int64_t>(i);
        scan.accept(t);
    }
    return -1;
}

// Reads OpenAI `response_format`. Returns false (with *err) on an unusable value.
bool parse_response_format(std::string_view raw, JsonSchema* schema, bool* active, const char** err) {
    *active = false;
    raw = trim_ascii_ws(raw);
    if (raw.empty() || raw == "null") return true;
    std::string_view type_raw;
    if (raw[0] != '{' || !extract_raw_field(raw, "type", &type_raw)) { *err = "response_format must be an object with a type"; return false; }
    type_raw = trim_ascii_ws(type_raw);
    if (type_raw == "\"text\"") return true;
    if (type_raw == "\"json_object\"") { schema->init_any_object(); *active = true; return true; }
    if (type_raw == "\"json_schema\"") {
        std::string_view js_raw, schema_raw;
        if (!extract_raw_field(raw, "json_schema", &js_raw) || !extract_raw_field(trim_ascii_ws(js_raw), "schema", &schema_raw)) {
            *err = "response_format.json_schema.schema is required";
            return false;
        }
        std::string msg;
        if (!schema->compile(trim_ascii_ws(schema_raw), &msg)) { *err = "response_format.json_schema.schema is not valid JSON"; return false; }
        *active = true;
        return true;
    }
    *err = "response_format.type must be text, json_object or json_schema";
    return false;
}

void handle_chat_completions(
    int fd, std::string_view body, Engine& engine, const Tokenizer& tokenizer,
    const ServerConfig& config, ServerBuffers& buffers, uint64_t& prng_state) {
    size_t pos = 0;
    skip_ws(body, pos);
    if (pos >= body.size() || body[pos] != '{') {
        send_error_json(fd, 400, "Invalid JSON body", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
    ++pos;

    std::string_view messages_json, tools_json;
    char req_model_buf[128], effort_buf[32], key_buf[128];
    std::string_view req_model = config.model_name;
    std::string_view effort = config.default_effort;
    const int32_t output_cap = static_cast<int32_t>(std::min(config.max_output_tokens, config.context_tokens_max));
    int32_t max_tokens = output_cap;
    bool has_max_completion_tokens = false;
    float req_temp = config.default_temperature;
    float req_top_p = config.default_top_p;
    int32_t req_top_k = config.default_top_k;
    bool has_seed = false;
    uint64_t req_seed = 0;
    bool stream = false;
    bool include_usage = false;
    bool enable_thinking = true;
    std::string_view response_format_raw;
    uint32_t max_reasoning_tokens = config.max_reasoning_tokens;
    bool has_max_reasoning = false;
    int32_t req_mtp = static_cast<int32_t>(config.default_mtp);
    StopList stops{};

    for (uint32_t fiter = 0; pos < body.size() && fiter < 64; ++fiter) {
        skip_ws(body, pos);
        if (pos < body.size() && body[pos] == '}') break;
        std::string_view k;
        if (!parse_json_string(body, pos, key_buf, sizeof(key_buf), &k)) {
            send_error_json(fd, 400, "Malformed JSON key", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
            return;
        }
        skip_ws(body, pos);
        if (pos >= body.size() || body[pos] != ':') {
            send_error_json(fd, 400, "Expected ':' in JSON", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
            return;
        }
        ++pos;
        skip_ws(body, pos);

        int64_t iv = 0;
        double dv = 0.0;
        bool bv = false;
        if (k == "messages") {
            const size_t start = pos; skip_json_value(body, pos); messages_json = body.substr(start, pos - start);
        } else if (k == "tools") {
            const size_t start = pos; skip_json_value(body, pos); tools_json = body.substr(start, pos - start);
        } else if (k == "model") {
            std::string_view ms;
            if (parse_json_string(body, pos, req_model_buf, sizeof(req_model_buf), &ms)) req_model = ms;
            else skip_json_value(body, pos);
        } else if (k == "reasoning_effort") {
            if (pos < body.size() && body[pos] == '"') {
                std::string_view es;
                if (parse_json_string(body, pos, effort_buf, sizeof(effort_buf), &es)) effort = es;
            } else {
                const size_t start = pos; skip_json_value(body, pos);
                const std::string_view rn = trim_ascii_ws(body.substr(start, pos - start));
                if (!rn.empty() && rn.size() <= sizeof(effort_buf) && rn != "null") {
                    std::memcpy(effort_buf, rn.data(), rn.size());
                    effort = std::string_view(effort_buf, rn.size());
                }
            }
        } else if (k == "max_completion_tokens") {
            if (parse_json_i64_opt(body, pos, &iv)) {
                max_tokens = clamp_i64_to_i32(iv, 1, output_cap);
                has_max_completion_tokens = true;
            }
        } else if (k == "max_tokens") {
            if (parse_json_i64_opt(body, pos, &iv) && !has_max_completion_tokens) {
                max_tokens = clamp_i64_to_i32(iv, 1, output_cap);
            }
        } else if (k == "temperature") {
            if (parse_json_f64_opt(body, pos, &dv)) req_temp = static_cast<float>(dv);
        } else if (k == "top_p") {
            if (parse_json_f64_opt(body, pos, &dv)) req_top_p = static_cast<float>(dv);
        } else if (k == "top_k") {
            if (parse_json_i64_opt(body, pos, &iv)) req_top_k = clamp_i64_to_i32(iv, 0, VOCAB_UNPADDED);
        } else if (k == "seed") {
            if (parse_json_i64_opt(body, pos, &iv)) { req_seed = static_cast<uint64_t>(iv); has_seed = true; }
        } else if (k == "stream") {
            stream = parse_json_bool(body, pos);
        } else if (k == "enable_thinking") {
            if (parse_json_bool_opt(body, pos, &bv)) enable_thinking = bv;
        } else if (k == "chat_template_kwargs") {
            const size_t start = pos; skip_json_value(body, pos);
            std::string_view et;
            if (extract_raw_field(body.substr(start, pos - start), "enable_thinking", &et)) {
                et = trim_ascii_ws(et);
                if (et == "false") enable_thinking = false;
                else if (et == "true") enable_thinking = true;
            }
        } else if (k == "max_reasoning_tokens") {
            if (parse_json_i64_opt(body, pos, &iv)) {
                max_reasoning_tokens = static_cast<uint32_t>(std::clamp<int64_t>(iv, 0, INT32_MAX));
                has_max_reasoning = true;
            }
        } else if (k == "response_format") {
            const size_t start = pos; skip_json_value(body, pos);
            response_format_raw = body.substr(start, pos - start);
        } else if (k == "stream_options") {
            const size_t start = pos; skip_json_value(body, pos);
            include_usage = extract_bool_field(body.substr(start, pos - start), "include_usage");
        } else if (k == "mtp") {
            if (parse_json_i64_opt(body, pos, &iv)) req_mtp = clamp_i64_to_i32(iv, 0, MTP_MAX_DRAFTS);
        } else if (k == "stop") {
            parse_stop_field(body, pos, stops);
        } else {
            skip_json_value(body, pos);
        }
        skip_ws(body, pos);
        if (pos < body.size() && body[pos] == ',') ++pos;
    }

    if (messages_json.size() < 2 || messages_json[0] != '[') {
        send_error_json(fd, 400, "'messages' must be a non-empty array", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
    if (max_tokens < 1) max_tokens = 1;
    if (!std::isfinite(req_temp) || req_temp < 0.0f) req_temp = 0.0f;
    if (!std::isfinite(req_top_p) || req_top_p <= 0.0f || req_top_p > 1.0f) req_top_p = 1.0f;
    if (req_top_k < 0) req_top_k = 0;
    req_mtp = std::clamp(req_mtp, 0, MTP_MAX_DRAFTS);

    JsonSchema json_schema;
    bool json_constrained = false;
    {
        const char* rf_err = nullptr;
        if (!parse_response_format(response_format_raw, &json_schema, &json_constrained, &rf_err)) {
            send_error_json(fd, 400, rf_err, "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
            return;
        }
    }

    RequestImages req_images{};
    const ChatEncodeResult enc = encode_chat_template_zero_alloc(
        tokenizer, messages_json, tools_json, effort, config.keep_history_thinking, enable_thinking, buffers, req_images);
    if (enc.image_error) {
        send_error_json(fd, 400, "Failed to fetch or decode image input (or exceeded 16 images)", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
    if (enc.overflow) {
        send_error_json(fd, 413, "Rendered prompt exceeds static buffer or max_context", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
    if (enc.prompt_len == 0 || enc.prompt_len >= config.context_tokens_max) {
        send_error_json(fd, 400, "Prompt empty or exceeds max_context", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }

    struct VisionEmbedsGuard {
        float* ptr = nullptr;
        ~VisionEmbedsGuard() {
            if (ptr) CUDA_CHECK(cudaFree(ptr));
        }
    } vis_guard{};
    Engine::VisionInput vision_in{};
    if (req_images.count > 0) {
        if (!engine.model || !engine.model->mw.vision.loaded) {
            send_error_json(fd, 400, "Loaded checkpoint does not include vision encoder weights", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
            return;
        }
        size_t total_vis_toks = 0;
        vis_guard.ptr = engine.encode_images(req_images.items, req_images.count, &total_vis_toks);
        if (!vis_guard.ptr || total_vis_toks == 0) {
            send_error_json(fd, 500, "Vision encoder failed to produce image embeddings", "server_error", buffers.response_buf, kResponseBufferBytes);
            return;
        }
        vision_in = {vis_guard.ptr, total_vis_toks};
    }
    const uint32_t gen_limit = std::min<uint32_t>(static_cast<uint32_t>(max_tokens),
                                                  config.context_tokens_max - static_cast<uint32_t>(enc.prompt_len));

    char req_id_buf[33];
    std::memcpy(req_id_buf, "chatcmpl-", 9);
    fill_random_hex(prng_state, req_id_buf + 9, 24);
    const std::string_view req_id(req_id_buf, 33);
    const int64_t created = epoch_seconds();

    InklingStreamParser parser{};
    parser.init(&tokenizer, config.raw_thinking, buffers, prng_state);
    if (!enable_thinking) parser.section = InklingSection::Text;
    prng_state = prng_state * 6364136223846793005ULL + 1;
    if (!has_seed) {
        req_seed = prng_state ^ clock_monotonic_ns();
        prng_state = prng_state * 6364136223846793005ULL + 1;
    }
    engine.sample_params = {req_temp, req_top_p, req_top_k, req_seed};

    struct ConstraintGuard {
        Engine& e;
        ~ConstraintGuard() { e.constraint = {}; }
    } constraint_guard{engine};
    JsonConstraintCtx json_ctx{};
    ToolSchemas tool_schemas;
    ToolCallConstraintCtx tool_ctx{};
    if (json_constrained) {
        json_ctx.init(&tokenizer, &json_schema, enable_thinking, engine.sample_params, buffers.order_buf);
        engine.constraint = {&JsonConstraintCtx::choose_cb, &json_ctx, buffers.logits_buf};
        req_mtp = 0;
    }
    // Tool calls are generated freely (with speculative drafts); a call whose arguments miss
    // its schema is regenerated from its <|content_invoke_tool_json|> token with the schema
    // enforced (see repair_tool_call below).
    // Default reasoning budget: the server's for low effort, four times that for medium and up.
    if (!has_max_reasoning && !(effort == "none" || effort == "minimal" || effort == "low")) {
        max_reasoning_tokens = static_cast<uint32_t>(std::min<uint64_t>(4ULL * max_reasoning_tokens, INT32_MAX));
    }
    LoopGuard loop_guard{};
    loop_guard.enabled = config.loop_guard;
    loop_guard.max_reasoning = max_reasoning_tokens;
    loop_guard.start(buffers.gen_tokens, 0);
    const bool check_tool_calls = !json_constrained && config.constrain_tool_calls && tools_json.size() > 2 &&
                                  tool_schemas.build(tools_json, buffers.scratch_str_buf, kScratchStringBytes);
    // Returns false if no repair was needed or possible. `reset_ctx` re-syncs the caller's
    // token callback after the parser is rebuilt from the kept tokens.
    auto repair_tool_call = [&](GenerateStats& stats, TokenCallbackFn cb, void* cb_ctx, auto&& reset_ctx) -> bool {
        if (!check_tool_calls) return false;
        const int64_t bad = find_invalid_tool_call(tokenizer, tool_schemas, buffers.gen_tokens,
                                                   stats.generated_tokens, buffers.order_buf);
        if (bad < 0) return false;
        const uint32_t keep = static_cast<uint32_t>(bad) + 1;
        const uint32_t len2 = static_cast<uint32_t>(enc.prompt_len) + keep;
        if (keep >= gen_limit || len2 >= config.context_tokens_max || len2 >= kPromptTokensMax) return false;
        std::memcpy(buffers.prompt_tokens + enc.prompt_len, buffers.gen_tokens, static_cast<size_t>(keep) * sizeof(int32_t));
        tool_ctx.init(&tokenizer, &tool_schemas, engine.sample_params, buffers.order_buf);
        for (uint32_t i = 0; i < keep; ++i) tool_ctx.accept(buffers.gen_tokens[i]);
        parser.init(&tokenizer, config.raw_thinking, buffers, prng_state);
        if (!enable_thinking) parser.section = InklingSection::Text;
        for (uint32_t i = 0; i < keep; ++i) (void)parser.feed(buffers.gen_tokens[i]);
        reset_ctx();
        engine.constraint = {&ToolCallConstraintCtx::choose_cb, &tool_ctx, buffers.logits_buf};
        loop_guard.start(buffers.gen_tokens, keep);
        loop_guard.max_reasoning = 0;
        const GenerateStats s2 = engine.generate_with_hint(
            buffers.prompt_tokens, len2, enc.system_prefix_tokens, gen_limit - keep, 0,
            buffers.gen_tokens + keep, gen_limit - keep, cb, cb_ctx, vision_in);
        engine.constraint = {};
        if (loop_guard.pending != LoopGuard::None) { loop_guard.stopped = true; loop_guard.pending = LoopGuard::None; }
        printf("[http] tool call arguments for '%.*s' missed their schema; regenerated %u tokens from token %u (%u replaced)\n",
               static_cast<int>(tool_ctx.header_len), tool_ctx.header, s2.generated_tokens, keep, tool_ctx.replaced);
        stats.generated_tokens = keep + s2.generated_tokens;
        stats.last_token_id = s2.last_token_id;
        stats.decode_ms += s2.prefill_ms + s2.decode_ms;
        return true;
    };

    // After the guard stopped generation: in reasoning, drop the repeats, close the reasoning
    // block and let the model go on (at most twice); otherwise the generation ends as "length".
    auto resolve_loops = [&](GenerateStats& stats, TokenCallbackFn cb, void* cb_ctx, auto&& reset_ctx) {
        for (int round = 0; loop_guard.pending != LoopGuard::None; ++round) {
            const LoopGuard::Act act = loop_guard.pending;
            const uint32_t keep = loop_guard.cut;
            printf("[http] %s: %s at token %u (period %u, reasoning %u tokens)\n",
                   act == LoopGuard::EndThinking && round < 2 ? "closing reasoning" : "stopping",
                   loop_guard.period > 0 ? "repetition loop" : "reasoning budget reached",
                   keep, loop_guard.period, loop_guard.reasoning_tokens);
            if (act != LoopGuard::EndThinking || round >= 2) { loop_guard.stopped = true; loop_guard.pending = LoopGuard::None; break; }
            const uint32_t len2 = static_cast<uint32_t>(enc.prompt_len) + keep + 1;
            if (keep + 1 >= gen_limit || len2 >= config.context_tokens_max || len2 >= kPromptTokensMax) {
                loop_guard.stopped = true; loop_guard.pending = LoopGuard::None; break;
            }
            buffers.gen_tokens[keep] = tokenizer.id_end_message;
            std::memcpy(buffers.prompt_tokens + enc.prompt_len, buffers.gen_tokens, static_cast<size_t>(keep + 1) * sizeof(int32_t));
            parser.init(&tokenizer, config.raw_thinking, buffers, prng_state);
            if (!enable_thinking) parser.section = InklingSection::Text;
            for (uint32_t i = 0; i <= keep; ++i) (void)parser.feed(buffers.gen_tokens[i]);
            reset_ctx();
            loop_guard.start(buffers.gen_tokens, keep + 1);
            loop_guard.reasoning_tokens = 0;
            loop_guard.max_reasoning = (round == 0 && max_reasoning_tokens > 0) ? max_reasoning_tokens / 2 : max_reasoning_tokens;
            const GenerateStats s2 = engine.generate_with_hint(
                buffers.prompt_tokens, len2, enc.system_prefix_tokens, gen_limit - keep - 1, static_cast<uint32_t>(req_mtp),
                buffers.gen_tokens + keep + 1, gen_limit - keep - 1, cb, cb_ctx, vision_in);
            stats.generated_tokens = keep + 1 + s2.generated_tokens;
            stats.last_token_id = s2.last_token_id;
            stats.decode_ms += s2.prefill_ms + s2.decode_ms;
        }
    };

    if (!stream) {
        struct NonStreamCtx {
            InklingStreamParser* parser;
            const StopList* stops;
            LoopGuard* guard;
            size_t scanned_len = 0;
            bool hit_stop = false;
            static bool on_token(void* p, int32_t tid) {
                auto* self = static_cast<NonStreamCtx*>(p);
                (void)self->parser->feed(tid);
                if (self->guard->observe(self->parser->section) != LoopGuard::None) return false;
                if (self->stops->count > 0) {
                    const std::string_view full = self->parser->content_writer.slice();
                    const size_t overlap = (self->stops->max_len > 1) ? (self->stops->max_len - 1) : 0;
                    const size_t scan_from = (self->scanned_len > overlap) ? (self->scanned_len - overlap) : 0;
                    size_t rel_idx = 0;
                    if (self->stops->find_first(full.substr(scan_from), &rel_idx)) {
                        self->parser->content_writer.length = scan_from + rel_idx;
                        self->hit_stop = true;
                        return false;
                    }
                    self->scanned_len = full.size();
                }
                return true;
            }
        } ctx{&parser, &stops, &loop_guard};

        GenerateStats stats = engine.generate_with_hint(
            buffers.prompt_tokens, enc.prompt_len, enc.system_prefix_tokens,
            gen_limit, static_cast<uint32_t>(req_mtp), buffers.gen_tokens, gen_limit, NonStreamCtx::on_token, &ctx, vision_in);
        if (!ctx.hit_stop) {
            resolve_loops(stats, NonStreamCtx::on_token, &ctx, [&] { ctx.scanned_len = 0; });
            if (!loop_guard.stopped) (void)repair_tool_call(stats, NonStreamCtx::on_token, &ctx, [&] { ctx.scanned_len = 0; });
        }

        const bool hit_eos = (stats.last_token_id == EOS_TOKEN);
        const char* finish_reason = (parser.tool_call_count > 0) ? "tool_calls"
                                  : loop_guard.stopped           ? "length"
                                  : (ctx.hit_stop || hit_eos)    ? "stop" : "length";
        const double dec_toks = static_cast<double>(stats.generated_tokens > 0 ? stats.generated_tokens - 1 : 0);
        const double tps = (stats.decode_ms > 0.0) ? (dec_toks * 1000.0 / stats.decode_ms) : 0.0;
        const double acc_pct = (stats.total_drafts > 0) ? (100.0 * static_cast<double>(stats.total_accepted) / stats.total_drafts) : 0.0;
        const double dl = (stats.spec_steps > 0) ? (dec_toks / stats.spec_steps) : 1.0;

        ByteWriter bw(buffers.response_buf, kResponseBufferBytes);
        bool ok = true;
        ok = ok && bw.try_append_slice("{\"id\":");
        ok = ok && bw.try_append_json_escaped(req_id);
        ok = ok && bw.try_append_fmt(",\"object\":\"chat.completion\",\"created\":%lld,\"model\":", static_cast<long long>(created));
        ok = ok && bw.try_append_json_escaped(req_model);
        ok = ok && bw.try_append_slice(",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        if (parser.content_writer.length > 0 || parser.tool_call_count == 0) {
            ok = ok && bw.try_append_json_escaped(parser.content_writer.slice());
        } else {
            ok = ok && bw.try_append_slice("null");
        }
        if (parser.reasoning_writer.length > 0) {
            ok = ok && bw.try_append_slice(",\"reasoning_content\":");
            ok = ok && bw.try_append_json_escaped(parser.reasoning_writer.slice());
        }
        if (parser.tool_call_count > 0) {
            ok = ok && bw.try_append_slice(",\"tool_calls\":[");
            for (size_t ti = 0; ti < parser.tool_call_count && ok; ++ti) {
                if (ti > 0) ok = ok && bw.try_append_byte(',');
                const ToolCallRecord& tc = parser.tool_calls[ti];
                std::string_view tname(parser.tool_store_writer.buffer + tc.name_offset, tc.name_len);
                std::string_view targs(parser.tool_store_writer.buffer + tc.args_offset, tc.args_len);
                ok = ok && bw.try_append_slice("{\"id\":\"call_");
                ok = ok && bw.try_append_slice(std::string_view(tc.id_hex, sizeof(tc.id_hex)));
                ok = ok && bw.try_append_slice("\",\"type\":\"function\",\"function\":{\"name\":");
                ok = ok && bw.try_append_json_escaped(tname);
                ok = ok && bw.try_append_slice(",\"arguments\":");
                ok = ok && bw.try_append_json_escaped(targs);
                ok = ok && bw.try_append_slice("}}");
            }
            ok = ok && bw.try_append_byte(']');
        }
        ok = ok && bw.try_append_slice("},\"finish_reason\":");
        ok = ok && bw.try_append_json_escaped(finish_reason);
        ok = ok && bw.try_append_fmt(
            "}],\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u,\"prompt_tokens_details\":{\"cached_tokens\":%u}},"
            "\"timings\":{\"prefill_ms\":%.2f,\"decode_ms\":%.2f,\"tokens_per_second\":%.2f,\"cached_tokens\":%u,\"spec_acc_pct\":%.1f,\"spec_dl\":%.2f}}",
            stats.prompt_tokens, stats.generated_tokens, stats.prompt_tokens + stats.generated_tokens, stats.cached_tokens,
            stats.prefill_ms, stats.decode_ms, tps, stats.cached_tokens, acc_pct, dl);
        if (!ok) {
            send_error_json(fd, 500, "Response exceeded 16 MiB static response buffer", "server_error", buffers.response_buf, kResponseBufferBytes);
            return;
        }
        printf("[http] POST /v1/chat/completions prompt=%u (cached=%u) gen=%u temp=%.2f prefill=%.1fms decode=%.1fms (%.1f tok/s, dl=%.2f)\n",
               stats.prompt_tokens, stats.cached_tokens, stats.generated_tokens, req_temp, stats.prefill_ms, stats.decode_ms, tps, dl);
        std::fflush(stdout);
        send_json_response(fd, 200, bw.slice());
        return;
    }

    // --- STREAMING SSE ---
    if (!send_sse_headers(fd)) return;
    const ByteWriter frame_prefix = chat_sse_frame_prefix(buffers.sse_buf, req_id, created, req_model);
    if (!send_chat_sse_delta(fd, frame_prefix, DeltaKind::Role, "assistant", nullptr, nullptr)) return;

    struct StreamCtx {
        int fd;
        ByteWriter frame_prefix;
        InklingStreamParser* parser;
        const StopList* stops;
        LoopGuard* guard;
        size_t emitted_len = 0;
        size_t scanned_len = 0;
        bool hit_stop = false;
        bool conn_ok = true;

        static bool on_token(void* p, int32_t tid) {
            auto* self = static_cast<StreamCtx*>(p);
            const auto delta = self->parser->feed(tid);
            const bool loop_hit = self->guard->observe(self->parser->section) != LoopGuard::None;
            if (!delta.reasoning.empty()) {
                if (!send_chat_sse_delta(self->fd, self->frame_prefix, DeltaKind::Reasoning, delta.reasoning, nullptr, nullptr)) {
                    self->conn_ok = false;
                    return false;
                }
            }
            if (!delta.content.empty()) {
                if (self->stops->count > 0) {
                    const std::string_view full = self->parser->content_writer.slice();
                    const size_t hold = (self->stops->max_len > 1) ? (self->stops->max_len - 1) : 0;
                    const size_t scan_from = (self->scanned_len > hold) ? (self->scanned_len - hold) : 0;
                    size_t rel_idx = 0;
                    if (self->stops->find_first(full.substr(scan_from), &rel_idx)) {
                        const size_t idx = scan_from + rel_idx;
                        if (idx > self->emitted_len) {
                            (void)send_chat_sse_delta(self->fd, self->frame_prefix, DeltaKind::Content,
                                                      full.substr(self->emitted_len, idx - self->emitted_len), nullptr, nullptr);
                        }
                        self->emitted_len = idx;
                        self->hit_stop = true;
                        return false;
                    }
                    self->scanned_len = full.size();
                    if (full.size() > self->emitted_len + hold) {
                        const size_t send_end = full.size() - hold;
                        if (!send_chat_sse_delta(self->fd, self->frame_prefix, DeltaKind::Content,
                                                 full.substr(self->emitted_len, send_end - self->emitted_len), nullptr, nullptr)) {
                            self->conn_ok = false;
                            return false;
                        }
                        self->emitted_len = send_end;
                    }
                } else {
                    if (!send_chat_sse_delta(self->fd, self->frame_prefix, DeltaKind::Content, delta.content, nullptr, nullptr)) {
                        self->conn_ok = false;
                        return false;
                    }
                    self->emitted_len = self->parser->content_writer.length;
                }
            }
            return !loop_hit;
        }
    } sctx{fd, frame_prefix, &parser, &stops, &loop_guard};

    GenerateStats stats = engine.generate_with_hint(
        buffers.prompt_tokens, enc.prompt_len, enc.system_prefix_tokens,
        gen_limit, static_cast<uint32_t>(req_mtp), buffers.gen_tokens, gen_limit, StreamCtx::on_token, &sctx, vision_in);
    if (!sctx.conn_ok) return;
    if (!sctx.hit_stop) {
        // Text already streamed stays sent; the rebuilt parser resumes after it.
        auto resync = [&] {
            sctx.emitted_len = std::min(sctx.emitted_len, parser.content_writer.length);
            sctx.scanned_len = std::min(sctx.scanned_len, parser.content_writer.length);
        };
        resolve_loops(stats, StreamCtx::on_token, &sctx, resync);
        if (!sctx.conn_ok) return;
        if (!loop_guard.stopped) (void)repair_tool_call(stats, StreamCtx::on_token, &sctx, resync);
        if (!sctx.conn_ok) return;
    }

    if (stops.count > 0 && !sctx.hit_stop && sctx.emitted_len < parser.content_writer.length) {
        const std::string_view rem = parser.content_writer.slice().substr(sctx.emitted_len);
        if (!send_chat_sse_delta(fd, frame_prefix, DeltaKind::Content, rem, nullptr, nullptr)) return;
    }
    if (parser.tool_call_count > 0) {
        if (!send_chat_sse_tool_calls(fd, frame_prefix, parser)) return;
    }

    const bool hit_eos = (stats.last_token_id == EOS_TOKEN);
    const char* finish_reason = (parser.tool_call_count > 0) ? "tool_calls"
                              : loop_guard.stopped           ? "length"
                              : (sctx.hit_stop || hit_eos)   ? "stop" : "length";
    (void)send_chat_sse_delta(fd, frame_prefix, DeltaKind::Empty, "", finish_reason, include_usage ? &stats : nullptr);
    (void)send_all(fd, "data: [DONE]\n\n");

    const double dec_toks = static_cast<double>(stats.generated_tokens > 0 ? stats.generated_tokens - 1 : 0);
    const double tps = (stats.decode_ms > 0.0) ? (dec_toks * 1000.0 / stats.decode_ms) : 0.0;
    const double dl = (stats.spec_steps > 0) ? (dec_toks / stats.spec_steps) : 1.0;
    printf("[http] POST /v1/chat/completions (stream) prompt=%u (cached=%u) gen=%u temp=%.2f prefill=%.1fms decode=%.1fms (%.1f tok/s, dl=%.2f)\n",
           stats.prompt_tokens, stats.cached_tokens, stats.generated_tokens, req_temp, stats.prefill_ms, stats.decode_ms, tps, dl);
    std::fflush(stdout);
}

void handle_text_completions(
    int fd, std::string_view body, Engine& engine, const Tokenizer& tokenizer,
    const ServerConfig& config, ServerBuffers& buffers, uint64_t& prng_state) {
    size_t pos = 0;
    skip_ws(body, pos);
    if (pos >= body.size() || body[pos] != '{') {
        send_error_json(fd, 400, "Invalid JSON body", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
    ++pos;

    std::string_view prompt;
    bool has_token_prompt = false;
    size_t token_prompt_len = 0;
    const int32_t output_cap = static_cast<int32_t>(std::min(config.max_output_tokens, config.context_tokens_max));
    int32_t max_tokens = output_cap;
    float req_temp = config.default_temperature;
    float req_top_p = config.default_top_p;
    int32_t req_top_k = config.default_top_k;
    bool has_seed = false;
    uint64_t req_seed = 0;
    bool stream = false;
    bool echo = false;
    int32_t req_logprobs = 0;
    bool return_tokens_as_token_ids = false;
    struct LogitBiasEntry { int32_t token_id; float bias; };
    LogitBiasEntry logit_biases[128]{};
    size_t logit_bias_count = 0;
    int32_t req_mtp = static_cast<int32_t>(config.default_mtp);
    char req_model_buf[128], key_buf[128];
    std::string_view req_model = config.model_name;
    StopList stops{};

    for (uint32_t fiter = 0; pos < body.size() && fiter < 64; ++fiter) {
        skip_ws(body, pos);
        if (pos < body.size() && body[pos] == '}') break;
        std::string_view k;
        if (!parse_json_string(body, pos, key_buf, sizeof(key_buf), &k)) break;
        skip_ws(body, pos);
        if (pos >= body.size() || body[pos] != ':') break;
        ++pos;
        skip_ws(body, pos);

        int64_t iv = 0;
        double dv = 0.0;
        bool bv = false;
        if (k == "prompt") {
            if (pos < body.size() && body[pos] == '"') {
                if (!parse_json_string(body, pos, buffers.render_buf, kPromptRenderBytes, &prompt)) prompt = {};
            } else if (pos < body.size() && body[pos] == '[') {
                ++pos;
                has_token_prompt = true;
                token_prompt_len = 0;
                while (pos < body.size()) {
                    skip_ws(body, pos);
                    if (pos >= body.size()) break;
                    if (body[pos] == ']') { ++pos; break; }
                    int64_t tid = 0;
                    if (parse_json_i64_opt(body, pos, &tid) && tid >= 0 && tid < VOCAB_UNPADDED) {
                        if (token_prompt_len < kPromptTokensMax) {
                            buffers.prompt_tokens[token_prompt_len++] = static_cast<int32_t>(tid);
                        }
                    }
                    skip_ws(body, pos);
                    if (pos < body.size() && body[pos] == ',') ++pos;
                }
            } else {
                skip_json_value(body, pos);
            }
        } else if (k == "model") {
            std::string_view ms;
            if (parse_json_string(body, pos, req_model_buf, sizeof(req_model_buf), &ms)) req_model = ms;
            else skip_json_value(body, pos);
        } else if (k == "max_tokens") {
            if (parse_json_i64_opt(body, pos, &iv)) max_tokens = clamp_i64_to_i32(iv, 1, output_cap);
        } else if (k == "temperature") {
            if (parse_json_f64_opt(body, pos, &dv)) req_temp = static_cast<float>(dv);
        } else if (k == "top_p") {
            if (parse_json_f64_opt(body, pos, &dv)) req_top_p = static_cast<float>(dv);
        } else if (k == "top_k") {
            if (parse_json_i64_opt(body, pos, &iv)) req_top_k = clamp_i64_to_i32(iv, 0, VOCAB_UNPADDED);
        } else if (k == "seed") {
            if (parse_json_i64_opt(body, pos, &iv)) { req_seed = static_cast<uint64_t>(iv); has_seed = true; }
        } else if (k == "stream") {
            stream = parse_json_bool(body, pos);
        } else if (k == "echo") {
            echo = parse_json_bool(body, pos);
        } else if (k == "logprobs") {
            if (pos < body.size() && (body[pos] == 't' || body[pos] == 'f')) {
                if (parse_json_bool_opt(body, pos, &bv)) req_logprobs = bv ? 1 : 0;
            } else if (parse_json_i64_opt(body, pos, &iv)) {
                req_logprobs = clamp_i64_to_i32(iv, 0, 64);
            }
        } else if (k == "return_tokens_as_token_ids") {
            return_tokens_as_token_ids = parse_json_bool(body, pos);
        } else if (k == "logit_bias") {
            if (pos < body.size() && body[pos] == '{') {
                ++pos;
                char bk_buf[32];
                for (uint32_t bi = 0; pos < body.size() && bi < 256; ++bi) {
                    skip_ws(body, pos);
                    if (pos >= body.size()) break;
                    if (body[pos] == '}') { ++pos; break; }
                    std::string_view bk;
                    if (!parse_json_string(body, pos, bk_buf, sizeof(bk_buf), &bk)) { skip_json_value(body, pos); break; }
                    skip_ws(body, pos);
                    if (pos < body.size() && body[pos] == ':') ++pos;
                    double bval = 0.0;
                    if (parse_json_f64_opt(body, pos, &bval) && logit_bias_count < 128) {
                        char tid_z[32];
                        if (!bk.empty() && bk.size() < sizeof(tid_z)) {
                            std::memcpy(tid_z, bk.data(), bk.size());
                            tid_z[bk.size()] = '\0';
                            char* endp = nullptr;
                            long long tid = std::strtoll(tid_z, &endp, 10);
                            if (endp == tid_z + bk.size() && tid >= 0 && tid < VOCAB_UNPADDED) {
                                logit_biases[logit_bias_count++] = {static_cast<int32_t>(tid), static_cast<float>(bval)};
                            }
                        }
                    }
                    skip_ws(body, pos);
                    if (pos < body.size() && body[pos] == ',') ++pos;
                }
            } else {
                skip_json_value(body, pos);
            }
        } else if (k == "mtp") {
            if (parse_json_i64_opt(body, pos, &iv)) req_mtp = clamp_i64_to_i32(iv, 0, MTP_MAX_DRAFTS);
        } else if (k == "stop") {
            parse_stop_field(body, pos, stops);
        } else {
            skip_json_value(body, pos);
        }
        skip_ws(body, pos);
        if (pos < body.size() && body[pos] == ',') ++pos;
    }

    size_t prompt_len = 0;
    if (has_token_prompt) {
        prompt_len = token_prompt_len;
    } else if (!prompt.empty()) {
        prompt_len = tokenizer.encode_into(prompt, true, buffers.tok_scratch, buffers.prompt_tokens, kPromptTokensMax);
    }
    if (prompt_len == 0) {
        send_error_json(fd, 400, "'prompt' must be a non-empty string or token ID array", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
    if (prompt_len >= config.context_tokens_max) {
        send_error_json(fd, 400, "Prompt exceeds max_context", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
    if (max_tokens < 1) max_tokens = 1;
    if (!std::isfinite(req_temp) || req_temp < 0.0f) req_temp = 0.0f;
    if (!std::isfinite(req_top_p) || req_top_p <= 0.0f || req_top_p > 1.0f) req_top_p = 1.0f;
    if (req_top_k < 0) req_top_k = 0;
    req_mtp = std::clamp(req_mtp, 0, MTP_MAX_DRAFTS);
    const uint32_t gen_limit = std::min<uint32_t>(static_cast<uint32_t>(max_tokens),
                                                  config.context_tokens_max - static_cast<uint32_t>(prompt_len));

    char req_id_buf[29];
    std::memcpy(req_id_buf, "cmpl-", 5);
    fill_random_hex(prng_state, req_id_buf + 5, 24);
    const std::string_view req_id(req_id_buf, 29);
    const int64_t created = epoch_seconds();

    InklingStreamParser parser{};
    parser.init(&tokenizer, true, buffers, prng_state);
    if (!has_seed) {
        req_seed = prng_state ^ clock_monotonic_ns();
        prng_state = prng_state * 6364136223846793005ULL + 1;
    }
    engine.sample_params = {req_temp, req_top_p, req_top_k, req_seed};

    if (!stream && gen_limit == 1 && (req_logprobs > 0 || logit_bias_count > 0)) {
        const GenerateStats stats = engine.generate_with_hint(
            buffers.prompt_tokens, prompt_len, 0, 1, 0, nullptr, 0, nullptr, nullptr, VisionInput{}, buffers.logits_buf);

        float* logits = buffers.logits_buf;
        for (size_t bi = 0; bi < logit_bias_count; ++bi) {
            logits[logit_biases[bi].token_id] += logit_biases[bi].bias;
        }
        const double inv_t = (req_temp >= 1e-4f) ? (1.0 / static_cast<double>(req_temp)) : 1.0;
        double max_l = -1e300;
        for (int v = 0; v < VOCAB_UNPADDED; ++v) {
            const double lv = static_cast<double>(logits[v]) * inv_t;
            if (lv > max_l) max_l = lv;
        }
        double sum_exp = 0.0;
        for (int v = 0; v < VOCAB_UNPADDED; ++v) {
            sum_exp += std::exp(static_cast<double>(logits[v]) * inv_t - max_l);
        }
        const double lse = max_l + std::log(sum_exp);

        const int k_top = std::clamp(req_logprobs > 0 ? req_logprobs : 1, 1, 64);
        int32_t top_ids[64];
        double top_vals[64];
        for (int i = 0; i < k_top; ++i) { top_ids[i] = 0; top_vals[i] = -1e300; }
        for (int v = 0; v < VOCAB_UNPADDED; ++v) {
            const double lv = static_cast<double>(logits[v]) * inv_t;
            if (lv > top_vals[k_top - 1]) {
                int j = k_top - 1;
                while (j > 0 && lv > top_vals[j - 1]) {
                    top_vals[j] = top_vals[j - 1];
                    top_ids[j] = top_ids[j - 1];
                    --j;
                }
                top_vals[j] = lv;
                top_ids[j] = v;
            }
        }
        const int32_t best_tid = top_ids[0];
        const double best_lp = top_vals[0] - lse;
        const std::string_view best_text = tokenizer.decode_token_bytes(best_tid, false);

        auto format_tok_key = [&](int32_t tid, char* buf, size_t cap) -> std::string_view {
            if (return_tokens_as_token_ids) {
                int n = std::snprintf(buf, cap, "token_id:%d", tid);
                return (n > 0 && static_cast<size_t>(n) < cap) ? std::string_view(buf, static_cast<size_t>(n)) : std::string_view{};
            }
            return tokenizer.decode_token_bytes(tid, false);
        };

        ByteWriter bw(buffers.response_buf, kResponseBufferBytes);
        bool ok = true;
        ok = ok && bw.try_append_slice("{\"id\":");
        ok = ok && bw.try_append_json_escaped(req_id);
        ok = ok && bw.try_append_fmt(",\"object\":\"text_completion\",\"created\":%lld,\"model\":", static_cast<long long>(created));
        ok = ok && bw.try_append_json_escaped(req_model);
        ok = ok && bw.try_append_slice(",\"choices\":[{\"index\":0,\"text\":");
        ok = ok && bw.try_append_json_escaped(best_text);
        if (req_logprobs > 0) {
            char tk_buf[32];
            ok = ok && bw.try_append_slice(",\"logprobs\":{\"text_offset\":[0],\"token_logprobs\":[");
            ok = ok && bw.try_append_fmt("%.10f", best_lp);
            ok = ok && bw.try_append_slice("],\"tokens\":[");
            ok = ok && bw.try_append_json_escaped(format_tok_key(best_tid, tk_buf, sizeof(tk_buf)));
            ok = ok && bw.try_append_slice("],\"top_logprobs\":[{");
            for (int i = 0; i < k_top && ok; ++i) {
                if (i > 0) ok = ok && bw.try_append_byte(',');
                ok = ok && bw.try_append_json_escaped(format_tok_key(top_ids[i], tk_buf, sizeof(tk_buf)));
                ok = ok && bw.try_append_fmt(":%.10f", top_vals[i] - lse);
            }
            ok = ok && bw.try_append_slice("}]}");
        } else {
            ok = ok && bw.try_append_slice(",\"logprobs\":null");
        }
        ok = ok && bw.try_append_slice(",\"finish_reason\":");
        ok = ok && bw.try_append_json_escaped((best_tid == EOS_TOKEN) ? "stop" : "length");
        ok = ok && bw.try_append_fmt(
            "}],\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":1,\"total_tokens\":%u},"
            "\"timings\":{\"prefill_ms\":%.2f,\"decode_ms\":0.00,\"tokens_per_second\":0.00}}",
            stats.prompt_tokens, stats.prompt_tokens + 1u, stats.prefill_ms);
        if (!ok) {
            send_error_json(fd, 500, "Response exceeded 16 MiB static response buffer", "server_error", buffers.response_buf, kResponseBufferBytes);
            return;
        }
        printf("[http] POST /v1/completions (logprobs=%d, bias=%zu) prompt=%u (cached=%u) prefill=%.1fms\n",
               req_logprobs, logit_bias_count, stats.prompt_tokens, stats.cached_tokens, stats.prefill_ms);
        std::fflush(stdout);
        send_json_response(fd, 200, bw.slice());
        return;
    }

    if (!stream) {
        struct CmplCtx {
            InklingStreamParser* parser;
            const StopList* stops;
            size_t scanned_len = 0;
            bool hit_stop = false;
            static bool on_token(void* p, int32_t tid) {
                auto* self = static_cast<CmplCtx*>(p);
                if (tid == self->parser->tokenizer->id_eos) return false;
                (void)self->parser->feed(tid);
                if (self->stops->count > 0) {
                    const std::string_view full = self->parser->content_writer.slice();
                    const size_t overlap = (self->stops->max_len > 1) ? (self->stops->max_len - 1) : 0;
                    const size_t scan_from = (self->scanned_len > overlap) ? (self->scanned_len - overlap) : 0;
                    size_t rel_idx = 0;
                    if (self->stops->find_first(full.substr(scan_from), &rel_idx)) {
                        self->parser->content_writer.length = scan_from + rel_idx;
                        self->hit_stop = true;
                        return false;
                    }
                    self->scanned_len = full.size();
                }
                return true;
            }
        } cctx{&parser, &stops};

        const GenerateStats stats = engine.generate(
            buffers.prompt_tokens, prompt_len, gen_limit, static_cast<uint32_t>(req_mtp), nullptr, 0, CmplCtx::on_token, &cctx);
        const bool hit_eos = (stats.last_token_id == EOS_TOKEN);
        const double dec_toks = static_cast<double>(stats.generated_tokens > 0 ? stats.generated_tokens - 1 : 0);
        const double tps = (stats.decode_ms > 0.0) ? (dec_toks * 1000.0 / stats.decode_ms) : 0.0;

        ByteWriter bw(buffers.response_buf, kResponseBufferBytes);
        bool ok = true;
        ok = ok && bw.try_append_slice("{\"id\":");
        ok = ok && bw.try_append_json_escaped(req_id);
        ok = ok && bw.try_append_fmt(",\"object\":\"text_completion\",\"created\":%lld,\"model\":", static_cast<long long>(created));
        ok = ok && bw.try_append_json_escaped(req_model);
        ok = ok && bw.try_append_slice(",\"choices\":[{\"index\":0,\"text\":");
        if (echo) {
            ByteWriter tw(buffers.scratch_str_buf, kScratchStringBytes);
            (void)tw.try_append_slice(prompt);
            (void)tw.try_append_slice(parser.content_writer.slice());
            ok = ok && bw.try_append_json_escaped(tw.slice());
        } else {
            ok = ok && bw.try_append_json_escaped(parser.content_writer.slice());
        }
        ok = ok && bw.try_append_slice(",\"logprobs\":null,\"finish_reason\":");
        ok = ok && bw.try_append_json_escaped((cctx.hit_stop || hit_eos) ? "stop" : "length");
        ok = ok && bw.try_append_fmt(
            "}],\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u},"
            "\"timings\":{\"prefill_ms\":%.2f,\"decode_ms\":%.2f,\"tokens_per_second\":%.2f}}",
            stats.prompt_tokens, stats.generated_tokens, stats.prompt_tokens + stats.generated_tokens,
            stats.prefill_ms, stats.decode_ms, tps);
        if (!ok) {
            send_error_json(fd, 500, "Response exceeded 16 MiB static response buffer", "server_error", buffers.response_buf, kResponseBufferBytes);
            return;
        }
        send_json_response(fd, 200, bw.slice());
        return;
    }

    if (!send_sse_headers(fd)) return;
    ByteWriter frame_prefix(buffers.sse_buf, kSseFrameBytes);
    frame_prefix.append_slice("data: {\"id\":");
    frame_prefix.append_json_escaped(req_id);
    frame_prefix.append_fmt(",\"object\":\"text_completion\",\"created\":%lld,\"model\":", static_cast<long long>(created));
    frame_prefix.append_json_escaped(req_model);
    frame_prefix.append_slice(",\"choices\":[{\"index\":0,\"text\":");

    struct StreamCmplCtx {
        int fd;
        ByteWriter frame_prefix;
        InklingStreamParser* parser;
        bool conn_ok = true;
        static bool on_token(void* p, int32_t tid) {
            auto* self = static_cast<StreamCmplCtx*>(p);
            if (tid == self->parser->tokenizer->id_eos) return false;
            const auto d = self->parser->feed(tid);
            if (!d.content.empty()) {
                ByteWriter bw = self->frame_prefix;
                bw.append_json_escaped(d.content);
                bw.append_slice(",\"logprobs\":null,\"finish_reason\":null}]}\n\n");
                if (!send_all(self->fd, bw.slice())) {
                    self->conn_ok = false;
                    return false;
                }
            }
            return true;
        }
    } scctx{fd, frame_prefix, &parser};

    const GenerateStats stats = engine.generate(
        buffers.prompt_tokens, prompt_len, gen_limit, static_cast<uint32_t>(req_mtp), nullptr, 0, StreamCmplCtx::on_token, &scctx);
    if (!scctx.conn_ok) return;
    ByteWriter bw = frame_prefix;
    bw.append_slice("\"\",\"logprobs\":null,\"finish_reason\":");
    bw.append_json_escaped((stats.last_token_id == EOS_TOKEN) ? "stop" : "length");
    bw.append_slice("}]}\n\n");
    (void)send_all(fd, bw.slice());
    (void)send_all(fd, "data: [DONE]\n\n");
}

void handle_tokenize(
    int fd, std::string_view body, const Tokenizer& tokenizer,
    const ServerConfig& config, ServerBuffers& buffers) {
    size_t pos = 0;
    skip_ws(body, pos);
    if (pos >= body.size() || body[pos] != '{') {
        send_error_json(fd, 400, "Invalid JSON body", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
    ++pos;

    std::string_view messages_json, tools_json, prompt_str;
    bool has_prompt = false;
    char effort_buf[32], key_buf[128];
    std::string_view effort = config.default_effort;

    for (uint32_t fiter = 0; pos < body.size() && fiter < 64; ++fiter) {
        skip_ws(body, pos);
        if (pos < body.size() && body[pos] == '}') break;
        std::string_view k;
        if (!parse_json_string(body, pos, key_buf, sizeof(key_buf), &k)) break;
        skip_ws(body, pos);
        if (pos >= body.size() || body[pos] != ':') break;
        ++pos;
        skip_ws(body, pos);

        if (k == "messages") {
            const size_t start = pos; skip_json_value(body, pos);
            messages_json = body.substr(start, pos - start);
        } else if (k == "tools") {
            const size_t start = pos; skip_json_value(body, pos);
            tools_json = body.substr(start, pos - start);
        } else if (k == "prompt" || k == "content") {
            if (pos < body.size() && body[pos] == '"') {
                if (parse_json_string(body, pos, buffers.render_buf, kPromptRenderBytes, &prompt_str)) has_prompt = true;
            } else {
                skip_json_value(body, pos);
            }
        } else if (k == "chat_template_kwargs") {
            const size_t start = pos; skip_json_value(body, pos);
            const std::string_view kwargs_obj = body.substr(start, pos - start);
            std::string_view eff_raw;
            if (extract_raw_field(kwargs_obj, "reasoning_effort", &eff_raw)) {
                if (!eff_raw.empty() && eff_raw[0] == '"') {
                    size_t ep = 0;
                    std::string_view es;
                    if (parse_json_string(eff_raw, ep, effort_buf, sizeof(effort_buf), &es)) effort = es;
                } else {
                    const std::string_view rn = trim_ascii_ws(eff_raw);
                    if (!rn.empty() && rn.size() <= sizeof(effort_buf) && rn != "null") {
                        std::memcpy(effort_buf, rn.data(), rn.size());
                        effort = std::string_view(effort_buf, rn.size());
                    }
                }
            }
        } else {
            skip_json_value(body, pos);
        }
        skip_ws(body, pos);
        if (pos < body.size() && body[pos] == ',') ++pos;
    }

    size_t tok_count = 0;
    if (!messages_json.empty() && messages_json[0] == '[') {
        RequestImages req_images{};
        const ChatEncodeResult enc = encode_chat_template_zero_alloc(
            tokenizer, messages_json, tools_json, effort, config.keep_history_thinking, true, buffers, req_images);
        if (enc.overflow || enc.image_error) {
            send_error_json(fd, 400, "Failed to tokenize messages", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
            return;
        }
        tok_count = enc.prompt_len;
    } else if (has_prompt) {
        tok_count = tokenizer.encode_into(prompt_str, true, buffers.tok_scratch, buffers.prompt_tokens, kPromptTokensMax);
    } else {
        send_error_json(fd, 400, "Missing 'prompt' or 'messages'", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }

    ByteWriter bw(buffers.response_buf, kResponseBufferBytes);
    bool ok = bw.try_append_fmt("{\"count\":%zu,\"max_model_len\":%u,\"tokens\":[", tok_count, config.context_tokens_max);
    for (size_t i = 0; i < tok_count && ok; ++i) {
        if (i > 0) ok = ok && bw.try_append_byte(',');
        ok = ok && bw.try_append_fmt("%d", buffers.prompt_tokens[i]);
    }
    ok = ok && bw.try_append_slice("]}");
    if (!ok) {
        send_error_json(fd, 500, "Tokenize response exceeded buffer", "server_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
    send_json_response(fd, 200, bw.slice());
}

void handle_detokenize(
    int fd, std::string_view body, const Tokenizer& tokenizer, ServerBuffers& buffers) {
    size_t pos = 0;
    skip_ws(body, pos);
    if (pos >= body.size() || body[pos] != '{') {
        send_error_json(fd, 400, "Invalid JSON body", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
    ++pos;

    size_t tok_count = 0;
    bool has_tokens = false;
    char key_buf[128];
    for (uint32_t fiter = 0; pos < body.size() && fiter < 32; ++fiter) {
        skip_ws(body, pos);
        if (pos < body.size() && body[pos] == '}') break;
        std::string_view k;
        if (!parse_json_string(body, pos, key_buf, sizeof(key_buf), &k)) break;
        skip_ws(body, pos);
        if (pos >= body.size() || body[pos] != ':') break;
        ++pos;
        skip_ws(body, pos);

        if (k == "tokens" && pos < body.size() && body[pos] == '[') {
            ++pos;
            has_tokens = true;
            while (pos < body.size()) {
                skip_ws(body, pos);
                if (pos >= body.size()) break;
                if (body[pos] == ']') { ++pos; break; }
                int64_t tid = 0;
                if (parse_json_i64_opt(body, pos, &tid) && tid >= 0 && tid < VOCAB_UNPADDED) {
                    if (tok_count < kPromptTokensMax) buffers.prompt_tokens[tok_count++] = static_cast<int32_t>(tid);
                }
                skip_ws(body, pos);
                if (pos < body.size() && body[pos] == ',') ++pos;
            }
        } else {
            skip_json_value(body, pos);
        }
        skip_ws(body, pos);
        if (pos < body.size() && body[pos] == ',') ++pos;
    }

    if (!has_tokens) {
        send_error_json(fd, 400, "Missing 'tokens' array", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }

    ByteWriter tw(buffers.render_buf, kPromptRenderBytes);
    for (size_t i = 0; i < tok_count; ++i) {
        const std::string_view piece = tokenizer.decode_token_bytes(buffers.prompt_tokens[i], false);
        if (!tw.try_append_slice(piece)) {
            send_error_json(fd, 500, "Detokenize output exceeded buffer", "server_error", buffers.response_buf, kResponseBufferBytes);
            return;
        }
    }
    ByteWriter bw(buffers.response_buf, kResponseBufferBytes);
    bool ok = bw.try_append_slice("{\"prompt\":");
    ok = ok && bw.try_append_json_escaped(tw.slice());
    ok = ok && bw.try_append_slice(",\"content\":");
    ok = ok && bw.try_append_json_escaped(tw.slice());
    ok = ok && bw.try_append_byte('}');
    if (!ok) {
        send_error_json(fd, 500, "Detokenize response exceeded buffer", "server_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
    send_json_response(fd, 200, bw.slice());
}

void sha256_hex(std::string_view data, char out_hex[65]) {
    static constexpr uint32_t kK[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    uint32_t h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    auto rotr = [](uint32_t x, uint32_t n) -> uint32_t { return (x >> n) | (x << (32u - n)); };
    auto compress = [&](const uint8_t blk[64]) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(blk[i * 4 + 0]) << 24) |
                   (static_cast<uint32_t>(blk[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(blk[i * 4 + 2]) << 8)  |
                   static_cast<uint32_t>(blk[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t t1 = hh + S1 + ch + kK[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    };

    const auto* ptr = reinterpret_cast<const uint8_t*>(data.data());
    const size_t len = data.size();
    size_t off = 0;
    while (len - off >= 64) {
        compress(ptr + off);
        off += 64;
    }
    uint8_t tail[128]{};
    const size_t rem = len - off;
    if (rem > 0) std::memcpy(tail, ptr + off, rem);
    tail[rem] = 0x80u;
    const size_t pad_len = (rem < 56) ? 64 : 128;
    const uint64_t bit_len = static_cast<uint64_t>(len) * 8ULL;
    for (int i = 0; i < 8; ++i) {
        tail[pad_len - 8 + i] = static_cast<uint8_t>((bit_len >> (56 - i * 8)) & 0xFFu);
    }
    compress(tail);
    if (pad_len == 128) compress(tail + 64);

    static constexpr char kHex[] = "0123456789abcdef";
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            out_hex[i * 8 + j] = kHex[(h[i] >> (28 - j * 4)) & 0xFu];
        }
    }
    out_hex[64] = '\0';
}

bool score_semif_row(
    std::string_view row_obj,
    Engine& engine,
    const Tokenizer& tokenizer,
    const ServerConfig& config,
    ServerBuffers& buffers,
    const int32_t slot_tids[16],
    ByteWriter& out_bw,
    const char** err_out) {
    static constexpr std::string_view kDirectSystem =
        "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
        "Respond with only its uppercase letter, with no explanation or reasoning.";
    static constexpr char kLetters[17] = "ABCDEFGHIJKLMNOP";

    const uint64_t t_row_start = clock_monotonic_ns();
    ByteWriter arena(buffers.content_buf, kContentBufferBytes);

    auto alloc_str = [&](std::string_view src_raw, std::string_view* dst) -> bool {
        if (src_raw.empty() || src_raw[0] != '"') return false;
        size_t p = 0;
        if (!parse_json_string(src_raw, p, arena.buffer + arena.length, arena.remaining(), dst)) return false;
        arena.length += dst->size();
        return true;
    };

    std::string_view id_raw, q_raw, state_raw, opts_raw;
    if (!extract_raw_field(row_obj, "id", &id_raw) ||
        !extract_raw_field(row_obj, "question", &q_raw) ||
        !extract_raw_field(row_obj, "state", &state_raw) ||
        !extract_raw_field(row_obj, "options", &opts_raw)) {
        *err_out = "row needs id, state, question and options";
        return false;
    }

    std::string_view row_id, question;
    if (!alloc_str(id_raw, &row_id) || row_id.empty() || !alloc_str(q_raw, &question) || question.empty()) {
        *err_out = "id and question must be nonempty strings";
        return false;
    }

    state_raw = trim_ascii_ws(state_raw);
    std::string_view state_str;
    bool state_is_str = false;
    if (!state_raw.empty() && state_raw[0] == '"') {
        if (!alloc_str(state_raw, &state_str) || state_str.empty()) {
            *err_out = "state must be a nonempty string, object, or array";
            return false;
        }
        state_is_str = true;
    } else if (!state_raw.empty() && (state_raw[0] == '{' || state_raw[0] == '[')) {
        size_t sp = 1;
        skip_ws(state_raw, sp);
        const char close_ch = (state_raw[0] == '{') ? '}' : ']';
        if (sp >= state_raw.size() || state_raw[sp] == close_ch) {
            *err_out = "state must be a nonempty string, object, or array";
            return false;
        }
    } else {
        *err_out = "state must be a nonempty string, object, or array";
        return false;
    }

    opts_raw = trim_ascii_ws(opts_raw);
    if (opts_raw.empty() || opts_raw[0] != '[') {
        *err_out = "options must contain 2-16 entries";
        return false;
    }

    std::string_view opt_ids[16];
    std::string_view opt_descs[16];
    size_t n_opts = 0;
    size_t op = 1;
    while (op < opts_raw.size()) {
        skip_ws(opts_raw, op);
        if (op >= opts_raw.size() || opts_raw[op] == ']') break;
        if (opts_raw[op] != '{' || n_opts >= 16) {
            *err_out = "options must contain 2-16 entries";
            return false;
        }
        const size_t ostart = op;
        skip_json_value(opts_raw, op);
        const std::string_view oobj = opts_raw.substr(ostart, op - ostart);
        std::string_view oid_raw, odesc_raw;
        if (!extract_raw_field(oobj, "id", &oid_raw) || !extract_raw_field(oobj, "description", &odesc_raw) ||
            !alloc_str(oid_raw, &opt_ids[n_opts]) || !alloc_str(odesc_raw, &opt_descs[n_opts])) {
            *err_out = "Each option needs string id and description fields";
            return false;
        }
        for (size_t prev = 0; prev < n_opts; ++prev) {
            if (opt_ids[prev] == opt_ids[n_opts]) {
                *err_out = "Option IDs must be unique";
                return false;
            }
        }
        ++n_opts;
        skip_ws(opts_raw, op);
        if (op < opts_raw.size() && opts_raw[op] == ',') ++op;
    }
    if (n_opts < 2 || n_opts > 16) {
        *err_out = "options must contain 2-16 entries";
        return false;
    }

    // Render exact direct-options-v1 prompt into buffers.render_buf (matches semif_bridge.py byte-for-byte)
    ByteWriter pw(buffers.render_buf, kPromptRenderBytes);
    bool pok = true;
    pok = pok && pw.try_append_slice("<|message_system|><|content_text|>");
    pok = pok && pw.try_append_slice(kDirectSystem);
    pok = pok && pw.try_append_slice("<|end_message|><|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>");

    // Payload JSON (matching Python json.dumps(payload, ensure_ascii=False))
    ByteWriter payload_w(buffers.scratch_str_buf, kScratchStringBytes);
    bool jok = true;
    jok = jok && payload_w.try_append_slice("{\"evidence\": ");
    if (state_is_str) {
        jok = jok && payload_w.try_append_json_escaped(state_str);
    } else {
        jok = jok && payload_w.try_append_slice(state_raw);
    }
    jok = jok && payload_w.try_append_slice(", \"criterion\": ");
    jok = jok && payload_w.try_append_json_escaped(question);
    jok = jok && payload_w.try_append_slice(", \"options\": [");
    for (size_t i = 0; i < n_opts && jok; ++i) {
        if (i > 0) jok = jok && payload_w.try_append_slice(", ");
        jok = jok && payload_w.try_append_slice("{\"letter\": \"");
        jok = jok && payload_w.try_append_byte(kLetters[i]);
        jok = jok && payload_w.try_append_slice("\", \"description\": ");
        jok = jok && payload_w.try_append_json_escaped(opt_descs[i]);
        jok = jok && payload_w.try_append_byte('}');
    }
    jok = jok && payload_w.try_append_slice("]}");

    pok = pok && jok && pw.try_append_slice(payload_w.slice());
    pok = pok && pw.try_append_slice("<|end_message|><|message_model|><|content_text|>");
    if (!pok) {
        *err_out = "Rendered SemIf prompt exceeded buffer";
        return false;
    }

    const std::string_view prompt_sv = pw.slice();
    char prompt_sha[65];
    sha256_hex(prompt_sv, prompt_sha);

    const size_t tok_count = tokenizer.encode_into(prompt_sv, true, buffers.tok_scratch, buffers.prompt_tokens, kPromptTokensMax);
    const uint32_t max_toks = std::min(config.semif_max_tokens, config.context_tokens_max - 1u);
    if (tok_count == 0 || tok_count > max_toks) {
        *err_out = "Input tokens exceed semif max_prompt_tokens limit; no truncation allowed";
        return false;
    }
    if (buffers.prompt_tokens[tok_count - 1] != tokenizer.id_text) {
        *err_out = "Answer boundary does not end on content_text token";
        return false;
    }

    const uint64_t t_fwd_start = clock_monotonic_ns();
    engine.sample_params = {1.0f, 1.0f, 0, 1};
    const GenerateStats stats = engine.generate_with_hint(
        buffers.prompt_tokens, tok_count, 0, 1, 0, nullptr, 0, nullptr, nullptr, VisionInput{}, buffers.logits_buf);
    const double forward_seconds = static_cast<double>(clock_monotonic_ns() - t_fwd_start) / 1e9;

    const float* logits = buffers.logits_buf;
    double max_l = -1e300;
    for (size_t i = 0; i < n_opts; ++i) {
        const double lv = static_cast<double>(logits[slot_tids[i]]);
        if (!std::isfinite(lv)) {
            *err_out = "Non-finite answer slot logit";
            return false;
        }
        if (lv > max_l) max_l = lv;
    }
    double probs[16];
    double sum_e = 0.0;
    for (size_t i = 0; i < n_opts; ++i) {
        probs[i] = std::exp(static_cast<double>(logits[slot_tids[i]]) - max_l);
        sum_e += probs[i];
    }
    if (!std::isfinite(sum_e) || sum_e <= 0.0) {
        *err_out = "answer slots carry no probability";
        return false;
    }
    for (size_t i = 0; i < n_opts; ++i) probs[i] /= sum_e;
    const double total_seconds = static_cast<double>(clock_monotonic_ns() - t_row_start) / 1e9;

    bool ok = true;
    ok = ok && out_bw.try_append_slice("{\"id\":");
    ok = ok && out_bw.try_append_json_escaped(row_id);
    ok = ok && out_bw.try_append_slice(",\"option_ids\":[");
    for (size_t i = 0; i < n_opts && ok; ++i) {
        if (i > 0) ok = ok && out_bw.try_append_byte(',');
        ok = ok && out_bw.try_append_json_escaped(opt_ids[i]);
    }
    ok = ok && out_bw.try_append_slice("],\"probabilities\":[");
    for (size_t i = 0; i < n_opts && ok; ++i) {
        if (i > 0) ok = ok && out_bw.try_append_byte(',');
        ok = ok && out_bw.try_append_fmt("%.17g", probs[i]);
    }
    ok = ok && out_bw.try_append_fmt(
        "],\"input_tokens\":%zu,\"forward_seconds\":%.9f,\"total_seconds\":%.9f,\"prompt_sha256\":\"%s\","
        "\"prompt_version\":\"direct-options-v1\",\"model\":{\"native_model\":\"Inkling-2-Mini\",\"weights_id\":\"%.*s\",\"backend\":\"squidDGX\",\"max_prompt_tokens\":%u},"
        "\"readout\":\"squidDGX last-position probabilities restricted to declared answer slots\","
        "\"probability_status\":\"conditional option score over quantized weights; uncalibrated as decision confidence\"}",
        tok_count, forward_seconds, total_seconds, prompt_sha,
        static_cast<int>(config.weights_id.size()), config.weights_id.data(), config.semif_max_tokens);
    if (!ok) {
        *err_out = "Score response exceeded buffer";
        return false;
    }

    printf("[http] POST /score id=%.*s prompt=%zu (cached=%u) fwd=%.1fms\n",
           static_cast<int>(row_id.size()), row_id.data(), tok_count, stats.cached_tokens, forward_seconds * 1000.0);
    std::fflush(stdout);
    return true;
}

void handle_semif_score(
    int fd, std::string_view body, Engine& engine, const Tokenizer& tokenizer,
    const ServerConfig& config, ServerBuffers& buffers) {
    static constexpr char kLetters[17] = "ABCDEFGHIJKLMNOP";
    int32_t slot_tids[16]{};
    for (int i = 0; i < 16; ++i) {
        int32_t tmp[4];
        const std::string_view letter(&kLetters[i], 1);
        const size_t n = tokenizer.encode_into(letter, false, buffers.tok_scratch, tmp, 4);
        if (n != 1 || tokenizer.decode_token_bytes(tmp[0], false) != letter) {
            ByteWriter eb(buffers.response_buf, kResponseBufferBytes);
            eb.append_slice("{\"error\":\"Answer slot letter is not a single exact round-trip token\"}");
            send_json_response(fd, 500, eb.slice());
            return;
        }
        slot_tids[i] = tmp[0];
    }

    size_t pos = 0;
    skip_ws(body, pos);
    if (pos >= body.size() || (body[pos] != '{' && body[pos] != '[')) {
        ByteWriter eb(buffers.response_buf, kResponseBufferBytes);
        eb.append_slice("{\"error\":\"ValueError: body must be a JSON object or array\"}");
        send_json_response(fd, 400, eb.slice());
        return;
    }

    ByteWriter bw(buffers.response_buf, kResponseBufferBytes);
    const char* err = nullptr;

    if (body[pos] == '{') {
        const size_t rstart = pos;
        skip_json_value(body, pos);
        if (!score_semif_row(body.substr(rstart, pos - rstart), engine, tokenizer, config, buffers, slot_tids, bw, &err)) {
            ByteWriter eb(buffers.response_buf, kResponseBufferBytes);
            eb.append_slice("{\"error\":");
            eb.append_json_escaped(err ? std::string_view(err) : std::string_view("Invalid score row"));
            eb.append_byte('}');
            send_json_response(fd, 400, eb.slice());
            return;
        }
        send_json_response(fd, 200, bw.slice());
        return;
    }

    ++pos;
    bool ok = bw.try_append_byte('[');
    size_t row_count = 0;
    while (pos < body.size() && ok) {
        skip_ws(body, pos);
        if (pos >= body.size() || body[pos] == ']') break;
        if (body[pos] != '{') {
            err = "Each element in score batch must be a JSON object";
            ok = false;
            break;
        }
        const size_t rstart = pos;
        skip_json_value(body, pos);
        if (row_count > 0) ok = ok && bw.try_append_byte(',');
        if (!ok || !score_semif_row(body.substr(rstart, pos - rstart), engine, tokenizer, config, buffers, slot_tids, bw, &err)) {
            ok = false;
            break;
        }
        ++row_count;
        skip_ws(body, pos);
        if (pos < body.size() && body[pos] == ',') ++pos;
    }
    ok = ok && bw.try_append_byte(']');
    if (!ok) {
        ByteWriter eb(buffers.response_buf, kResponseBufferBytes);
        eb.append_slice("{\"error\":");
        eb.append_json_escaped(err ? std::string_view(err) : std::string_view("Invalid score batch"));
        eb.append_byte('}');
        send_json_response(fd, 400, eb.slice());
        return;
    }
    send_json_response(fd, 200, bw.slice());
}

// --- Connection handling ----------------------------------------------------------------------
// One engine, many connections: each accepted connection gets a worker thread and a slot that
// owns its request buffer and a small reply buffer. Cheap endpoints (/health, /v1/models,
// OPTIONS, errors) answer at once from the slot; engine endpoints take g_engine_mutex and use the
// shared ServerBuffers, so they run one at a time in arrival order (std::mutex is not FIFO, but
// the router sends one request per target at a time). With every slot taken, a new connection
// gets an immediate 503 instead of waiting unanswered in the listen backlog.
constexpr int kConnSlots = 8;
constexpr size_t kSmallBufBytes = 64 * 1024;

struct ConnSlot {
    char* request_buf = nullptr;
    char* small_buf = nullptr;
    bool in_use = false;
};

std::mutex g_engine_mutex;
std::atomic<bool> g_engine_busy{false};
std::atomic<int> g_engine_waiting{0};

struct EngineLock {
    std::unique_lock<std::mutex> lk;
    EngineLock() {
        g_engine_waiting.fetch_add(1, std::memory_order_relaxed);
        lk = std::unique_lock<std::mutex>(g_engine_mutex);
        g_engine_waiting.fetch_sub(1, std::memory_order_relaxed);
        g_engine_busy.store(true, std::memory_order_relaxed);
    }
    ~EngineLock() { g_engine_busy.store(false, std::memory_order_relaxed); }
};

void handle_connection(
    int fd, Engine& engine, const Tokenizer& tokenizer,
    const ServerConfig& config, ServerBuffers& buffers, ConnSlot& slot, uint64_t& prng_state, bool is_semif_port) {
    const HttpReadResult rr = read_http_request(fd, slot.request_buf, kRequestBufferBytes);
    if (rr.status == HttpReadStatus::BadRequest) {
        send_error_json(fd, 400, rr.err_msg, "invalid_request_error", slot.small_buf, kSmallBufBytes);
        return;
    }
    if (rr.status == HttpReadStatus::PayloadTooLarge) {
        send_error_json(fd, 413, "Request body exceeds 32 MiB static HTTP buffer", "invalid_request_error", slot.small_buf, kSmallBufBytes);
        return;
    }
    if (rr.status == HttpReadStatus::NotImplemented) {
        send_error_json(fd, 501, rr.err_msg, "invalid_request_error", slot.small_buf, kSmallBufBytes);
        return;
    }
    if (rr.status != HttpReadStatus::Ok) return;
    const ParsedHttp& req = rr.req;

    if (req.method == "OPTIONS") {
        (void)send_all(fd,
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Content-Length: 0\r\nConnection: close\r\n\r\n");
        return;
    }

    if (is_semif_port) {
        if (req.method == "GET" && req.path == "/health") {
            ByteWriter bw(slot.small_buf, kSmallBufBytes);
            bw.append_fmt("{\"native_model\":\"Inkling-2-Mini\",\"weights_id\":\"%.*s\",\"backend\":\"squidDGX\",\"port\":%u,"
                          "\"prompt_version\":\"direct-options-v1\",\"model\":{\"native_model\":\"Inkling-2-Mini\",\"weights_id\":\"%.*s\",\"backend\":\"squidDGX\"}}",
                          static_cast<int>(config.weights_id.size()), config.weights_id.data(), config.semif_port,
                          static_cast<int>(config.weights_id.size()), config.weights_id.data());
            send_json_response(fd, 200, bw.slice());
            return;
        }
        if (req.method == "POST" && req.path == "/score") {
            EngineLock engine_lock;
        handle_semif_score(fd, req.body, engine, tokenizer, config, buffers);
            return;
        }
        ByteWriter bw(slot.small_buf, kSmallBufBytes);
        bw.append_slice("{\"error\":\"not found\"}");
        send_json_response(fd, 404, bw.slice());
        return;
    }

    if (req.method == "GET" && req.path == "/health") {
        ByteWriter bw(slot.small_buf, kSmallBufBytes);
        bw.append_slice("{\"status\":\"ok\",\"model\":");
        bw.append_json_escaped(config.model_name);
        bw.append_fmt(",\"max_context\":%u,\"max_output_tokens\":%u,\"mtp\":%u,\"engine\":\"squidDGX\",\"busy\":%s,\"queued\":%d}",
                      config.context_tokens_max, config.max_output_tokens, config.default_mtp,
                      g_engine_busy.load(std::memory_order_relaxed) ? "true" : "false",
                      g_engine_waiting.load(std::memory_order_relaxed));
        send_json_response(fd, 200, bw.slice());
        return;
    }

    if (!config.api_key.empty()) {
        static constexpr std::string_view kPrefix = "Bearer ";
        const bool authorized = req.auth_header.size() >= kPrefix.size() &&
                                req.auth_header.substr(0, kPrefix.size()) == kPrefix &&
                                constant_time_eql(req.auth_header.substr(kPrefix.size()), config.api_key);
        if (!authorized) {
            send_error_json(fd, 401, "Invalid API key", "authentication_error", slot.small_buf, kSmallBufBytes);
            return;
        }
    }

    if (req.method == "GET" && (req.path == "/v1/models" || req.path == "/models")) {
        ByteWriter bw(slot.small_buf, kSmallBufBytes);
        bw.append_slice("{\"object\":\"list\",\"data\":[{\"id\":");
        bw.append_json_escaped(config.model_name);
        bw.append_fmt(",\"object\":\"model\",\"created\":%lld,\"owned_by\":\"squidDGX\",\"max_model_len\":%u,\"max_output_tokens\":%u}]}",
                      static_cast<long long>(epoch_seconds()), config.context_tokens_max, config.max_output_tokens);
        send_json_response(fd, 200, bw.slice());
        return;
    }

    if (req.method == "POST" && (req.path == "/v1/chat/completions" || req.path == "/chat/completions")) {
        EngineLock engine_lock;
        handle_chat_completions(fd, req.body, engine, tokenizer, config, buffers, prng_state);
        return;
    }

    if (req.method == "POST" && (req.path == "/v1/completions" || req.path == "/completions")) {
        EngineLock engine_lock;
        handle_text_completions(fd, req.body, engine, tokenizer, config, buffers, prng_state);
        return;
    }

    if (req.method == "POST" && (req.path == "/tokenize" || req.path == "/v1/tokenize")) {
        EngineLock engine_lock;
        handle_tokenize(fd, req.body, tokenizer, config, buffers);
        return;
    }

    if (req.method == "POST" && (req.path == "/detokenize" || req.path == "/v1/detokenize")) {
        EngineLock engine_lock;
        handle_detokenize(fd, req.body, tokenizer, buffers);
        return;
    }

    if (req.method == "POST" && req.path == "/score") {
        EngineLock engine_lock;
        handle_semif_score(fd, req.body, engine, tokenizer, config, buffers);
        return;
    }

    send_error_json(fd, 404, "Unknown endpoint", "invalid_request_error", slot.small_buf, kSmallBufBytes);
}

} // namespace

namespace {

bool file_fingerprint_hex(const char* path_z, char out_hex[65]) {
    const int fd = ::open(path_z, O_RDONLY);
    if (fd < 0) return false;
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) { ::close(fd); return false; }
    const size_t bytes = static_cast<size_t>(st.st_size);
    void* map = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) return false;
    ::madvise(map, bytes, MADV_SEQUENTIAL);
    const char* base = static_cast<const char*>(map);

    constexpr size_t kChunk = 64ull << 20;
    const size_t n_chunks = (bytes + kChunk - 1) / kChunk;
    std::vector<char> digests(n_chunks * 64);
    std::atomic<size_t> next{0};
    const unsigned n_threads = std::max(1u, std::min<unsigned>(std::thread::hardware_concurrency(), 16u));
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (unsigned t = 0; t < n_threads; ++t) {
        workers.emplace_back([&] {
            char hex[65];
            for (size_t i = next.fetch_add(1); i < n_chunks; i = next.fetch_add(1)) {
                const size_t off = i * kChunk;
                sha256_hex(std::string_view(base + off, std::min(kChunk, bytes - off)), hex);
                std::memcpy(digests.data() + i * 64, hex, 64);
            }
        });
    }
    for (auto& w : workers) w.join();
    ::munmap(map, bytes);
    sha256_hex(std::string_view(digests.data(), digests.size()), out_hex);
    return true;
}

} // namespace

bool compute_weights_id(std::string_view model_directory, std::string_view ablation_path,
                        std::string_view mode, char* out, size_t out_cap) {
    char path_buf[4096];
    char main_hex[65];
    if (!file_fingerprint_hex(format_path_z(path_buf, sizeof(path_buf), model_directory, "/model-00001-of-00001.safetensors"), main_hex))
        return false;
    int n = std::snprintf(out, out_cap, "Inkling-2-Mini.sha256x64m-%.16s.%.*s", main_hex,
                          static_cast<int>(mode.size()), mode.data());
    if (n < 0 || static_cast<size_t>(n) >= out_cap) return false;
    if (!ablation_path.empty()) {
        char abl_hex[65];
        if (!file_fingerprint_hex(format_path_z(path_buf, sizeof(path_buf), ablation_path, ""), abl_hex)) return false;
        const int m = std::snprintf(out + n, out_cap - static_cast<size_t>(n), ".abl-sha256x64m-%.16s", abl_hex);
        if (m < 0 || static_cast<size_t>(n + m) >= out_cap) return false;
    }
    return true;
}

size_t valid_utf8_prefix_len(std::string_view s) {
    const size_t n = s.size();
    if (n == 0) return 0;
    const size_t check = std::min<size_t>(n, 3);
    for (size_t back = 1; back <= check; ++back) {
        const uint8_t c = static_cast<uint8_t>(s[n - back]);
        if ((c & 0x80) == 0) return n;
        if ((c & 0xC0) == 0xC0) {
            const size_t expected = ((c & 0xE0) == 0xC0) ? 2
                                  : ((c & 0xF0) == 0xE0) ? 3
                                  : ((c & 0xF8) == 0xF0) ? 4 : 1;
            return (back < expected) ? (n - back) : n;
        }
    }
    return n;
}

int run_openai_server(Engine& engine, const Tokenizer& tokenizer, const ServerConfig& config) {
    SQUID_ASSERT(config.port > 0);
    SQUID_ASSERT(config.context_tokens_max >= 64);

    std::signal(SIGPIPE, SIG_IGN);
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    auto bind_listener = [&](uint16_t bind_port) -> int {
        int sfd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (sfd < 0) return -1;
        int reuse = 1;
        (void)::setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(bind_port);
        addr.sin_addr.s_addr = INADDR_ANY;
        if (config.host != "0.0.0.0") {
            char host_z[128];
            SQUID_ASSERT(config.host.size() < sizeof(host_z));
            std::memcpy(host_z, config.host.data(), config.host.size());
            host_z[config.host.size()] = '\0';
            if (::inet_pton(AF_INET, host_z, &addr.sin_addr) != 1) {
                fprintf(stderr, "[server] Invalid bind host '%.*s'\n", static_cast<int>(config.host.size()), config.host.data());
                ::close(sfd);
                return -1;
            }
        }
        if (::bind(sfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            fprintf(stderr, "[server] Failed to bind %.*s:%u\n", static_cast<int>(config.host.size()), config.host.data(), bind_port);
            ::close(sfd);
            return -1;
        }
        if (::listen(sfd, 128) < 0) {
            fprintf(stderr, "[server] Failed to listen on port %u\n", bind_port);
            ::close(sfd);
            return -1;
        }
        return sfd;
    };

    int listen_fd = bind_listener(config.port);
    if (listen_fd < 0) return 1;

    int semif_fd = -1;
    if (config.semif_port > 0 && config.semif_port != config.port) {
        semif_fd = bind_listener(config.semif_port);
        if (semif_fd < 0) {
            ::close(listen_fd);
            return 1;
        }
    }

    g_stop_requested.store(false, std::memory_order_release);
    g_listen_fd.store(listen_fd, std::memory_order_release);
    g_semif_fd.store(semif_fd, std::memory_order_release);
    engine.enable_prefix_cache(config.prefix_sessions);

    ServerBuffers buffers{};
    buffers.init();

    uint64_t prng_state = clock_monotonic_ns() ^ 0x9e3779b97f4a7c15ULL;
    if (prng_state == 0) prng_state = 0x9e3779b97f4a7c15ULL;

    printf("[server] squidDGX C++20 OpenAI API listening on http://%.*s:%u/v1 (model=%.*s, ctx=%u, max_out=%u, mtp=%u, zero-malloc=on)\n",
           static_cast<int>(config.host.size()), config.host.data(), config.port,
           static_cast<int>(config.model_name.size()), config.model_name.data(),
           config.context_tokens_max, config.max_output_tokens, config.default_mtp);
    if (semif_fd >= 0) {
        printf("[server] SemIf decision endpoint listening natively on http://%.*s:%u/score (max_prompt_tokens=%u, no sidecar)\n",
               static_cast<int>(config.host.size()), config.host.data(), config.semif_port, config.semif_max_tokens);
    }
    std::fflush(stdout);

    ConnSlot slots[kConnSlots];
    std::mutex slots_mu;
    std::atomic<int> active_conns{0};
    for (auto& sl : slots) {
        sl.request_buf = alloc_host_zeroed<char>(kRequestBufferBytes);
        sl.small_buf = alloc_host_zeroed<char>(kSmallBufBytes);
    }

    timeval sock_timeout{};
    sock_timeout.tv_sec = 30;
    sock_timeout.tv_usec = 0;

    while (!g_stop_requested.load(std::memory_order_acquire)) {
        pollfd pfds[2]{};
        nfds_t nfds = 0;
        pfds[nfds++] = {listen_fd, POLLIN, 0};
        if (semif_fd >= 0) pfds[nfds++] = {semif_fd, POLLIN, 0};

        int ready = ::poll(pfds, nfds, -1);
        if (ready < 0) {
            if (g_stop_requested.load(std::memory_order_acquire)) break;
            if (errno == EINTR) continue;
            continue;
        }
        for (nfds_t i = 0; i < nfds; ++i) {
            if (!(pfds[i].revents & POLLIN)) continue;
            int client_fd = ::accept(pfds[i].fd, nullptr, nullptr);
            if (client_fd < 0) continue;
            int nodelay = 1;
            (void)::setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
            (void)::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &sock_timeout, sizeof(sock_timeout));
            (void)::setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &sock_timeout, sizeof(sock_timeout));

            ConnSlot* slot = nullptr;
            {
                std::lock_guard<std::mutex> g(slots_mu);
                for (auto& sl : slots) if (!sl.in_use) { sl.in_use = true; slot = &sl; break; }
            }
            if (!slot) {
                (void)send_all(client_fd,
                    "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nRetry-After: 1\r\n"
                    "Content-Length: 72\r\nConnection: close\r\n\r\n"
                    "{\"error\":{\"message\":\"server busy\",\"type\":\"server_error\",\"code\":\"busy\"}}\n");
                ::close(client_fd);
                continue;
            }
            const bool on_semif = (pfds[i].fd == semif_fd);
            active_conns.fetch_add(1, std::memory_order_acq_rel);
            std::thread([&, client_fd, slot, on_semif] {
                handle_connection(client_fd, engine, tokenizer, config, buffers, *slot, prng_state, on_semif);
                ::close(client_fd);
                {
                    std::lock_guard<std::mutex> g(slots_mu);
                    slot->in_use = false;
                }
                active_conns.fetch_sub(1, std::memory_order_acq_rel);
            }).detach();
        }
    }

    // Let in-flight connections finish (they reference the buffers below); give up after 30 s.
    for (int w = 0; w < 300 && active_conns.load(std::memory_order_acquire) > 0; ++w) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (active_conns.load(std::memory_order_acquire) == 0) {
        buffers.deinit();
        for (auto& sl : slots) { free_host(sl.request_buf); free_host(sl.small_buf); }
    }
    int rem_fd = g_listen_fd.exchange(-1, std::memory_order_acq_rel);
    if (rem_fd >= 0) ::close(rem_fd);
    int rem_sfd = g_semif_fd.exchange(-1, std::memory_order_acq_rel);
    if (rem_sfd >= 0) ::close(rem_sfd);
    printf("[server] Shutdown complete.\n");
    std::fflush(stdout);
    return 0;
}

} // namespace squid
