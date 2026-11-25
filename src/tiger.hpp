#pragma once

// TigerStyle core primitives for pure C++20 squidDGX:
// Always-on release assertions, monotonic timing, ARM64 NEON JSON scanning,
// fixed-capacity ByteWriter, and direct POSIX I/O. Zero dynamic alloc on hot paths.

#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string_view>
#include <unistd.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace squid {

[[noreturn]] inline void fail_assert(const char* expr, const char* file, int line) {
    fprintf(stderr, "SQUID_ASSERT failed at %s:%d: %s\n", file, line, expr);
    std::abort();
}

#define SQUID_ASSERT(cond)                                    \
    do {                                                      \
        if (__builtin_expect(!(cond), 0)) {                   \
            ::squid::fail_assert(#cond, __FILE__, __LINE__);  \
        }                                                     \
    } while (0)

inline uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
    SQUID_ASSERT(alignment > 0);
    SQUID_ASSERT((alignment & (alignment - 1)) == 0);
    const uint64_t mask = alignment - 1;
    const uint64_t result = (value + mask) & ~mask;
    SQUID_ASSERT(result >= value);
    SQUID_ASSERT((result & mask) == 0);
    return result;
}

inline uint64_t hash_bytes_u64(std::string_view bytes) {
    uint64_t hash = 14695981039346656037ULL;
    const auto* p = reinterpret_cast<const uint8_t*>(bytes.data());
    for (size_t i = 0; i < bytes.size(); ++i) {
        hash ^= static_cast<uint64_t>(p[i]);
        hash *= 1099511628211ULL;
    }
    return hash == 0 ? 1ULL : hash;
}

inline uint64_t hash_pair_u64(uint64_t key) {
    uint64_t mixed = key + 0x9e3779b97f4a7c15ULL;
    mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebULL;
    mixed = mixed ^ (mixed >> 31);
    return mixed == 0 ? 1ULL : mixed;
}

inline uint64_t clock_monotonic_ns() {
    struct timespec ts{};
    const int rc = clock_gettime(CLOCK_MONOTONIC, &ts);
    SQUID_ASSERT(rc == 0);
    SQUID_ASSERT(ts.tv_sec >= 0 && ts.tv_nsec >= 0 && ts.tv_nsec < 1000000000L);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

inline int64_t epoch_seconds() {
    struct timespec ts{};
    const int rc = clock_gettime(CLOCK_REALTIME, &ts);
    SQUID_ASSERT(rc == 0);
    SQUID_ASSERT(ts.tv_sec >= 0);
    return static_cast<int64_t>(ts.tv_sec);
}

template <typename T>
inline T* alloc_host_zeroed(size_t count) {
    SQUID_ASSERT(count > 0);
    void* raw = std::calloc(count, sizeof(T));
    SQUID_ASSERT(raw != nullptr);
    return static_cast<T*>(raw);
}

inline void free_host(void* ptr) {
    std::free(ptr);
}

// Scan ordinary JSON string bytes 16 at a time with ARM64 NEON.
template <bool EscapeControls>
inline size_t json_plain_prefix(std::string_view bytes) {
    const auto* p = reinterpret_cast<const uint8_t*>(bytes.data());
    const size_t n = bytes.size();
    size_t i = 0;
    const size_t head = n < 4 ? n : 4;
    for (; i < head; ++i) {
        const uint8_t c = p[i];
        if (c == '"' || c == '\\' || (EscapeControls && c < 0x20)) return i;
    }
#if defined(__aarch64__)
    const uint8x16_t v_quote = vdupq_n_u8('"');
    const uint8x16_t v_slash = vdupq_n_u8('\\');
    const uint8x16_t v_ctrl  = vdupq_n_u8(0x20);
    while (n - i >= 16) {
        uint8x16_t blk = vld1q_u8(p + i);
        uint8x16_t hit = vorrq_u8(vceqq_u8(blk, v_quote), vceqq_u8(blk, v_slash));
        if constexpr (EscapeControls) {
            hit = vorrq_u8(hit, vcltq_u8(blk, v_ctrl));
        }
        if (vmaxvq_u8(hit) != 0) break;
        i += 16;
    }
#endif
    for (; i < n; ++i) {
        const uint8_t c = p[i];
        if (c == '"' || c == '\\' || (EscapeControls && c < 0x20)) break;
    }
    return i;
}

struct ByteWriter {
    char* buffer = nullptr;
    size_t capacity = 0;
    size_t length = 0;

    ByteWriter() = default;
    ByteWriter(char* buf, size_t cap) : buffer(buf), capacity(cap), length(0) {
        SQUID_ASSERT(buf != nullptr && cap > 0);
    }

    void reset() {
        SQUID_ASSERT(length <= capacity);
        length = 0;
    }

    std::string_view slice() const {
        SQUID_ASSERT(length <= capacity);
        return std::string_view(buffer, length);
    }

    size_t remaining() const {
        SQUID_ASSERT(length <= capacity);
        return capacity - length;
    }

    void append_slice(std::string_view text) {
        SQUID_ASSERT(length <= capacity);
        SQUID_ASSERT(text.size() <= capacity - length);
        if (!text.empty()) std::memcpy(buffer + length, text.data(), text.size());
        length += text.size();
    }

    bool try_append_slice(std::string_view text) {
        SQUID_ASSERT(length <= capacity);
        if (text.size() > capacity - length) return false;
        if (!text.empty()) std::memcpy(buffer + length, text.data(), text.size());
        length += text.size();
        return true;
    }

    void append_byte(char byte) {
        SQUID_ASSERT(length < capacity);
        buffer[length++] = byte;
    }

    bool try_append_byte(char byte) {
        SQUID_ASSERT(length <= capacity);
        if (length >= capacity) return false;
        buffer[length++] = byte;
        return true;
    }

    __attribute__((format(printf, 2, 3)))
    bool try_append_fmt(const char* fmt, ...) {
        SQUID_ASSERT(length <= capacity);
        const size_t avail = capacity - length;
        if (avail == 0) return false;
        va_list ap;
        va_start(ap, fmt);
        int written = std::vsnprintf(buffer + length, avail, fmt, ap);
        va_end(ap);
        if (written < 0 || static_cast<size_t>(written) >= avail) return false;
        length += static_cast<size_t>(written);
        return true;
    }

    __attribute__((format(printf, 2, 3)))
    void append_fmt(const char* fmt, ...) {
        SQUID_ASSERT(length <= capacity);
        const size_t avail = capacity - length;
        SQUID_ASSERT(avail > 0);
        va_list ap;
        va_start(ap, fmt);
        int written = std::vsnprintf(buffer + length, avail, fmt, ap);
        va_end(ap);
        SQUID_ASSERT(written >= 0 && static_cast<size_t>(written) < avail);
        length += static_cast<size_t>(written);
    }

    bool try_append_json_escaped(std::string_view text) {
        SQUID_ASSERT(length <= capacity);
        const size_t saved = length;
        if (!try_append_byte('"')) return false;
        static constexpr char kHex[] = "0123456789abcdef";
        std::string_view rest = text;
        while (!rest.empty()) {
            const size_t plain_len = json_plain_prefix<true>(rest);
            if (!try_append_slice(rest.substr(0, plain_len))) {
                length = saved;
                return false;
            }
            if (plain_len == rest.size()) break;
            const uint8_t byte = static_cast<uint8_t>(rest[plain_len]);
            rest.remove_prefix(plain_len + 1);
            bool ok = false;
            switch (byte) {
                case '"':  ok = try_append_slice("\\\""); break;
                case '\\': ok = try_append_slice("\\\\"); break;
                case '\n': ok = try_append_slice("\\n"); break;
                case '\r': ok = try_append_slice("\\r"); break;
                case '\t': ok = try_append_slice("\\t"); break;
                case 0x08: ok = try_append_slice("\\b"); break;
                case 0x0c: ok = try_append_slice("\\f"); break;
                default: {
                    char esc[6] = {'\\', 'u', '0', '0', kHex[byte >> 4], kHex[byte & 0x0f]};
                    ok = try_append_slice(std::string_view(esc, 6));
                    break;
                }
            }
            if (!ok) {
                length = saved;
                return false;
            }
        }
        if (!try_append_byte('"')) {
            length = saved;
            return false;
        }
        return true;
    }

    void append_json_escaped(std::string_view text) {
        SQUID_ASSERT(try_append_json_escaped(text));
    }
};

inline bool write_fd_all_checked(int fd, std::string_view bytes) {
    SQUID_ASSERT(fd >= 0);
    size_t sent = 0;
    const size_t max_iter = bytes.size() + 64;
    for (size_t iter = 0; sent < bytes.size() && iter < max_iter; ++iter) {
        ssize_t n = ::write(fd, bytes.data() + sent, bytes.size() - sent);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return sent == bytes.size();
}

inline void write_fd_all(int fd, std::string_view bytes) {
    (void)write_fd_all_checked(fd, bytes);
}

inline const char* format_path_z(char* buf, size_t buf_cap, std::string_view dir, std::string_view suffix) {
    SQUID_ASSERT(!dir.empty());
    SQUID_ASSERT(buf_cap > dir.size() + suffix.size() + 1);
    std::memcpy(buf, dir.data(), dir.size());
    std::memcpy(buf + dir.size(), suffix.data(), suffix.size());
    buf[dir.size() + suffix.size()] = '\0';
    return buf;
}

} // namespace squid
