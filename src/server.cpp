#include "server.hpp"

#include "tiger.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

std::atomic<bool> g_stop_requested{false};
std::atomic<int> g_listen_fd{-1};

void handle_signal(int) {
    g_stop_requested.store(true, std::memory_order_release);
    int fd = g_listen_fd.exchange(-1, std::memory_order_acq_rel);
    if (fd >= 0) ::close(fd);
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
    char* request_buf = nullptr;
    char* render_buf = nullptr;
    char* response_buf = nullptr;
    char* content_buf = nullptr;
    char* reasoning_buf = nullptr;
    char* tool_raw_buf = nullptr;
    char* tool_store_buf = nullptr;
    char* scratch_str_buf = nullptr;
    char* sse_buf = nullptr;
    int32_t* prompt_tokens = nullptr;
    TokenizerScratch tok_scratch{};

    void init() {
        request_buf = alloc_host_zeroed<char>(kRequestBufferBytes);
        render_buf = alloc_host_zeroed<char>(kPromptRenderBytes);
        response_buf = alloc_host_zeroed<char>(kResponseBufferBytes);
        content_buf = alloc_host_zeroed<char>(kContentBufferBytes);
        reasoning_buf = alloc_host_zeroed<char>(kReasoningBufferBytes);
        tool_raw_buf = alloc_host_zeroed<char>(kToolRawBytes);
        tool_store_buf = alloc_host_zeroed<char>(kToolStoreBytes);
        scratch_str_buf = alloc_host_zeroed<char>(kScratchStringBytes);
        sse_buf = alloc_host_zeroed<char>(kSseFrameBytes);
        prompt_tokens = alloc_host_zeroed<int32_t>(kPromptTokensMax);
        tok_scratch.init();
    }

    void deinit() {
        free_host(request_buf);
        free_host(render_buf);
        free_host(response_buf);
        free_host(content_buf);
        free_host(reasoning_buf);
        free_host(tool_raw_buf);
        free_host(tool_store_buf);
        free_host(scratch_str_buf);
        free_host(sse_buf);
        free_host(prompt_tokens);
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

struct ChatEncodeResult {
    size_t prompt_len = 0;
    uint32_t system_prefix_tokens = 0;
    bool overflow = false;
};

ChatEncodeResult encode_chat_template_zero_alloc(
    const Tokenizer& tokenizer,
    std::string_view messages_json,
    std::string_view tools_json,
    std::string_view effort,
    bool keep_history_thinking,
    bool enable_thinking,
    ServerBuffers& buffers) {
    size_t out_len = 0;
    bool overflow = false;
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
                        std::string_view ts;
                        if (extract_string_field(c_raw.substr(ps_start, ap - ps_start), "text", scratch, kScratchStringBytes, &ts)) {
                            emitted_assistant_block = true;
                            append_encoded_segment(tokenizer, role_tok, true, buffers, out_len, overflow);
                            if (has_resolved_name) append_encoded_segment(tokenizer, resolved_name, false, buffers, out_len, overflow);
                            append_encoded_segment(tokenizer, "<|content_text|>", true, buffers, out_len, overflow);
                            append_encoded_segment(tokenizer, ts, false, buffers, out_len, overflow);
                            append_encoded_segment(tokenizer, "<|end_message|>", true, buffers, out_len, overflow);
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
    return {out_len, system_prefix_tokens, overflow};
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

    const ChatEncodeResult enc = encode_chat_template_zero_alloc(
        tokenizer, messages_json, tools_json, effort, config.keep_history_thinking, enable_thinking, buffers);
    if (enc.overflow) {
        send_error_json(fd, 413, "Rendered prompt exceeds static buffer or max_context", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
    if (enc.prompt_len == 0 || enc.prompt_len >= config.context_tokens_max) {
        send_error_json(fd, 400, "Prompt empty or exceeds max_context", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
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

    if (!stream) {
        struct NonStreamCtx {
            InklingStreamParser* parser;
            const StopList* stops;
            size_t scanned_len = 0;
            bool hit_stop = false;
            static bool on_token(void* p, int32_t tid) {
                auto* self = static_cast<NonStreamCtx*>(p);
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
        } ctx{&parser, &stops};

        const GenerateStats stats = engine.generate_with_hint(
            buffers.prompt_tokens, enc.prompt_len, enc.system_prefix_tokens,
            gen_limit, static_cast<uint32_t>(req_mtp), nullptr, 0, NonStreamCtx::on_token, &ctx);

        const bool hit_eos = (stats.last_token_id == EOS_TOKEN);
        const char* finish_reason = (parser.tool_call_count > 0) ? "tool_calls"
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
            "}}],\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u,\"prompt_tokens_details\":{\"cached_tokens\":%u}},"
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
        size_t emitted_len = 0;
        size_t scanned_len = 0;
        bool hit_stop = false;
        bool conn_ok = true;

        static bool on_token(void* p, int32_t tid) {
            auto* self = static_cast<StreamCtx*>(p);
            const auto delta = self->parser->feed(tid);
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
            return true;
        }
    } sctx{fd, frame_prefix, &parser, &stops};

    const GenerateStats stats = engine.generate_with_hint(
        buffers.prompt_tokens, enc.prompt_len, enc.system_prefix_tokens,
        gen_limit, static_cast<uint32_t>(req_mtp), nullptr, 0, StreamCtx::on_token, &sctx);
    if (!sctx.conn_ok) return;

    if (stops.count > 0 && !sctx.hit_stop && sctx.emitted_len < parser.content_writer.length) {
        const std::string_view rem = parser.content_writer.slice().substr(sctx.emitted_len);
        if (!send_chat_sse_delta(fd, frame_prefix, DeltaKind::Content, rem, nullptr, nullptr)) return;
    }
    if (parser.tool_call_count > 0) {
        if (!send_chat_sse_tool_calls(fd, frame_prefix, parser)) return;
    }

    const bool hit_eos = (stats.last_token_id == EOS_TOKEN);
    const char* finish_reason = (parser.tool_call_count > 0) ? "tool_calls"
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
    const int32_t output_cap = static_cast<int32_t>(std::min(config.max_output_tokens, config.context_tokens_max));
    int32_t max_tokens = output_cap;
    float req_temp = config.default_temperature;
    float req_top_p = config.default_top_p;
    int32_t req_top_k = config.default_top_k;
    bool has_seed = false;
    uint64_t req_seed = 0;
    bool stream = false;
    bool echo = false;
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
        if (k == "prompt") {
            if (pos < body.size() && body[pos] == '"') {
                if (!parse_json_string(body, pos, buffers.render_buf, kPromptRenderBytes, &prompt)) prompt = {};
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

    if (prompt.empty()) {
        send_error_json(fd, 400, "'prompt' must be a non-empty string", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
    if (max_tokens < 1) max_tokens = 1;
    if (!std::isfinite(req_temp) || req_temp < 0.0f) req_temp = 0.0f;
    if (!std::isfinite(req_top_p) || req_top_p <= 0.0f || req_top_p > 1.0f) req_top_p = 1.0f;
    if (req_top_k < 0) req_top_k = 0;
    req_mtp = std::clamp(req_mtp, 0, MTP_MAX_DRAFTS);

    const size_t prompt_len = tokenizer.encode_into(prompt, true, buffers.tok_scratch, buffers.prompt_tokens, kPromptTokensMax);
    if (prompt_len == 0 || prompt_len >= config.context_tokens_max) {
        send_error_json(fd, 400, "Prompt exceeds max_context", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
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
            "}}],\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u},"
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

void handle_connection(
    int fd, Engine& engine, const Tokenizer& tokenizer,
    const ServerConfig& config, ServerBuffers& buffers, uint64_t& prng_state) {
    const HttpReadResult rr = read_http_request(fd, buffers.request_buf, kRequestBufferBytes);
    if (rr.status == HttpReadStatus::BadRequest) {
        send_error_json(fd, 400, rr.err_msg, "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
    if (rr.status == HttpReadStatus::PayloadTooLarge) {
        send_error_json(fd, 413, "Request body exceeds 32 MiB static HTTP buffer", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
        return;
    }
    if (rr.status == HttpReadStatus::NotImplemented) {
        send_error_json(fd, 501, rr.err_msg, "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
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

    if (req.method == "GET" && req.path == "/health") {
        ByteWriter bw(buffers.response_buf, kResponseBufferBytes);
        bw.append_slice("{\"status\":\"ok\",\"model\":");
        bw.append_json_escaped(config.model_name);
        bw.append_fmt(",\"max_context\":%u,\"max_output_tokens\":%u,\"mtp\":%u,\"engine\":\"squidDGX\"}",
                      config.context_tokens_max, config.max_output_tokens, config.default_mtp);
        send_json_response(fd, 200, bw.slice());
        return;
    }

    if (!config.api_key.empty()) {
        static constexpr std::string_view kPrefix = "Bearer ";
        const bool authorized = req.auth_header.size() >= kPrefix.size() &&
                                req.auth_header.substr(0, kPrefix.size()) == kPrefix &&
                                constant_time_eql(req.auth_header.substr(kPrefix.size()), config.api_key);
        if (!authorized) {
            send_error_json(fd, 401, "Invalid API key", "authentication_error", buffers.response_buf, kResponseBufferBytes);
            return;
        }
    }

    if (req.method == "GET" && (req.path == "/v1/models" || req.path == "/models")) {
        ByteWriter bw(buffers.response_buf, kResponseBufferBytes);
        bw.append_slice("{\"object\":\"list\",\"data\":[{\"id\":");
        bw.append_json_escaped(config.model_name);
        bw.append_fmt(",\"object\":\"model\",\"created\":%lld,\"owned_by\":\"squidDGX\",\"max_model_len\":%u,\"max_output_tokens\":%u}]}",
                      static_cast<long long>(epoch_seconds()), config.context_tokens_max, config.max_output_tokens);
        send_json_response(fd, 200, bw.slice());
        return;
    }

    if (req.method == "POST" && (req.path == "/v1/chat/completions" || req.path == "/chat/completions")) {
        handle_chat_completions(fd, req.body, engine, tokenizer, config, buffers, prng_state);
        return;
    }

    if (req.method == "POST" && (req.path == "/v1/completions" || req.path == "/completions")) {
        handle_text_completions(fd, req.body, engine, tokenizer, config, buffers, prng_state);
        return;
    }

    send_error_json(fd, 404, "Unknown endpoint", "invalid_request_error", buffers.response_buf, kResponseBufferBytes);
}

} // namespace

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

    int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        fprintf(stderr, "[server] Failed to create socket\n");
        return 1;
    }

    int reuse = 1;
    (void)::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(config.port);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (config.host != "0.0.0.0") {
        char host_z[128];
        SQUID_ASSERT(config.host.size() < sizeof(host_z));
        std::memcpy(host_z, config.host.data(), config.host.size());
        host_z[config.host.size()] = '\0';
        if (::inet_pton(AF_INET, host_z, &addr.sin_addr) != 1) {
            fprintf(stderr, "[server] Invalid bind host '%.*s'\n", static_cast<int>(config.host.size()), config.host.data());
            ::close(listen_fd);
            return 1;
        }
    }

    if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        fprintf(stderr, "[server] Failed to bind %.*s:%u\n", static_cast<int>(config.host.size()), config.host.data(), config.port);
        ::close(listen_fd);
        return 1;
    }
    if (::listen(listen_fd, 128) < 0) {
        fprintf(stderr, "[server] Failed to listen\n");
        ::close(listen_fd);
        return 1;
    }

    g_stop_requested.store(false, std::memory_order_release);
    g_listen_fd.store(listen_fd, std::memory_order_release);
    engine.enable_prefix_cache();

    ServerBuffers buffers{};
    buffers.init();

    uint64_t prng_state = clock_monotonic_ns() ^ 0x9e3779b97f4a7c15ULL;
    if (prng_state == 0) prng_state = 0x9e3779b97f4a7c15ULL;

    printf("[server] squidDGX C++20 OpenAI API listening on http://%.*s:%u/v1 (model=%.*s, ctx=%u, max_out=%u, mtp=%u, zero-malloc=on)\n",
           static_cast<int>(config.host.size()), config.host.data(), config.port,
           static_cast<int>(config.model_name.size()), config.model_name.data(),
           config.context_tokens_max, config.max_output_tokens, config.default_mtp);
    std::fflush(stdout);

    timeval sock_timeout{};
    sock_timeout.tv_sec = 30;
    sock_timeout.tv_usec = 0;

    while (!g_stop_requested.load(std::memory_order_acquire)) {
        int client_fd = ::accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            if (g_stop_requested.load(std::memory_order_acquire)) break;
            if (errno == EINTR) continue;
            continue;
        }
        int nodelay = 1;
        (void)::setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
        (void)::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &sock_timeout, sizeof(sock_timeout));
        (void)::setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &sock_timeout, sizeof(sock_timeout));

        handle_connection(client_fd, engine, tokenizer, config, buffers, prng_state);
        ::close(client_fd);
    }

    buffers.deinit();
    int rem_fd = g_listen_fd.exchange(-1, std::memory_order_acq_rel);
    if (rem_fd >= 0) ::close(rem_fd);
    printf("[server] Shutdown complete.\n");
    std::fflush(stdout);
    return 0;
}

} // namespace squid
