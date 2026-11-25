#include "safetensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>

namespace squid {

namespace {

struct Span {
    size_t start;
    size_t length;
};

inline void st_skip_ws(std::string_view data, size_t& pos) {
    while (pos < data.size()) {
        const char c = data[pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos;
        else break;
    }
}

Span st_parse_raw_string(std::string_view data, size_t& pos) {
    st_skip_ws(data, pos);
    SQUID_ASSERT(pos < data.size() && data[pos] == '"');
    ++pos;
    const size_t start = pos;
    while (pos < data.size()) {
        const char c = data[pos];
        if (c == '\\') {
            pos = (pos + 2 <= data.size()) ? (pos + 2) : data.size();
            continue;
        }
        if (c == '"') {
            const size_t end = pos;
            ++pos;
            return {start, end - start};
        }
        ++pos;
    }
    SQUID_ASSERT(false);
    return {0, 0};
}

uint64_t st_parse_u64(std::string_view data, size_t& pos) {
    st_skip_ws(data, pos);
    SQUID_ASSERT(pos < data.size());
    uint64_t value = 0;
    uint32_t digits = 0;
    for (; pos < data.size() && digits < 20; ++digits) {
        const char c = data[pos];
        if (c < '0' || c > '9') break;
        value = value * 10 + static_cast<uint64_t>(c - '0');
        ++pos;
    }
    SQUID_ASSERT(digits > 0);
    return value;
}

void st_skip_value(std::string_view data, size_t& pos) {
    st_skip_ws(data, pos);
    SQUID_ASSERT(pos < data.size());
    const char first = data[pos];
    if (first == '"') {
        (void)st_parse_raw_string(data, pos);
        return;
    }
    if (first == '{' || first == '[') {
        int depth = 0;
        while (pos < data.size()) {
            const char c = data[pos];
            if (c == '"') {
                (void)st_parse_raw_string(data, pos);
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
        SQUID_ASSERT(false);
    }
    while (pos < data.size()) {
        const char c = data[pos];
        if (c == ',' || c == '}' || c == ']' || c == ' ' || c == '\t' || c == '\n' || c == '\r') break;
        ++pos;
    }
}

} // namespace

bool SafetensorsFile::open_file(const char* path_z) {
    SQUID_ASSERT(mapped_ptr == nullptr);
    SQUID_ASSERT(file_descriptor < 0);

    int fd = ::open(path_z, O_RDONLY);
    if (fd < 0) return false;

    off_t sz = ::lseek(fd, 0, SEEK_END);
    if (sz <= 8) {
        ::close(fd);
        return false;
    }
    const uint64_t fbytes = static_cast<uint64_t>(sz);
    void* raw = ::mmap(nullptr, static_cast<size_t>(fbytes), PROT_READ, MAP_PRIVATE, fd, 0);
    if (raw == MAP_FAILED) {
        ::close(fd);
        return false;
    }

    const auto* base = static_cast<const uint8_t*>(raw);
    uint64_t hlen = 0;
    std::memcpy(&hlen, base, sizeof(uint64_t));
    SQUID_ASSERT(hlen > 0 && hlen <= SAFETENSORS_HEADER_MAX);
    SQUID_ASSERT(8 + hlen <= fbytes);

    file_descriptor = fd;
    mapped_ptr = base;
    file_bytes = fbytes;
    header_bytes = hlen;
    data_base_offset = 8 + hlen;
    tensor_count = 0;
    for (uint32_t i = 0; i < SAFETENSORS_HASH_SLOTS; ++i) hash_table[i] = kEmptySlot;

    parse_header(std::string_view(reinterpret_cast<const char*>(base + 8), static_cast<size_t>(hlen)));
    SQUID_ASSERT(tensor_count > 0);
    return true;
}

void SafetensorsFile::close_file() {
    if (mapped_ptr) {
        SQUID_ASSERT(file_bytes > 0);
        ::munmap(const_cast<uint8_t*>(mapped_ptr), static_cast<size_t>(file_bytes));
        mapped_ptr = nullptr;
    }
    if (file_descriptor >= 0) {
        // Do NOT call posix_fadvise(POSIX_FADV_DONTNEED): evicting the 30 GB model from
        // the OS page cache regresses warm startup from ~2.1s to ~20s.
        ::close(file_descriptor);
        file_descriptor = -1;
    }
    file_bytes = 0;
    tensor_count = 0;
}

const TensorInfo* SafetensorsFile::lookup_optional(std::string_view name) const {
    SQUID_ASSERT(mapped_ptr != nullptr);
    SQUID_ASSERT(!name.empty());
    const std::string_view header(reinterpret_cast<const char*>(mapped_ptr + 8), static_cast<size_t>(header_bytes));
    const uint64_t hash = hash_bytes_u64(name);
    const uint32_t mask = SAFETENSORS_HASH_SLOTS - 1;
    uint32_t slot = static_cast<uint32_t>(hash & mask);

    for (uint32_t probe = 0; probe < SAFETENSORS_HASH_SLOTS; ++probe) {
        const uint16_t idx = hash_table[slot];
        if (idx == kEmptySlot) break;
        SQUID_ASSERT(idx < tensor_count);
        const TensorInfo& info = tensors[idx];
        if (header.substr(info.name_offset, info.name_length) == name) {
            return &info;
        }
        slot = (slot + 1) & mask;
    }
    return nullptr;
}

const TensorInfo* SafetensorsFile::lookup(std::string_view name) const {
    if (const TensorInfo* info = lookup_optional(name)) return info;
    fprintf(stderr, "[safetensors] Missing tensor: %.*s\n", static_cast<int>(name.size()), name.data());
    std::abort();
}

void SafetensorsFile::insert_tensor(std::string_view header, const TensorInfo& info) {
    SQUID_ASSERT(tensor_count < SAFETENSORS_MAX_TENSORS);
    SQUID_ASSERT(info.name_length > 0);
    const uint16_t idx = static_cast<uint16_t>(tensor_count);
    tensors[idx] = info;
    tensor_count += 1;

    const std::string_view name = header.substr(info.name_offset, info.name_length);
    const uint32_t mask = SAFETENSORS_HASH_SLOTS - 1;
    uint32_t slot = static_cast<uint32_t>(hash_bytes_u64(name) & mask);
    for (uint32_t probe = 0; probe < SAFETENSORS_HASH_SLOTS; ++probe) {
        if (hash_table[slot] == kEmptySlot) {
            hash_table[slot] = idx;
            return;
        }
        slot = (slot + 1) & mask;
    }
    SQUID_ASSERT(false);
}

void SafetensorsFile::parse_header(std::string_view header) {
    SQUID_ASSERT(header.size() >= 2);
    size_t pos = 0;
    st_skip_ws(header, pos);
    SQUID_ASSERT(pos < header.size() && header[pos] == '{');
    ++pos;

    const uint32_t max_entries = SAFETENSORS_MAX_TENSORS + 16;
    for (uint32_t entry_iter = 0; pos < header.size() && entry_iter < max_entries; ++entry_iter) {
        st_skip_ws(header, pos);
        if (pos < header.size() && header[pos] == '}') break;

        const Span key_span = st_parse_raw_string(header, pos);
        st_skip_ws(header, pos);
        SQUID_ASSERT(pos < header.size() && header[pos] == ':');
        ++pos;
        st_skip_ws(header, pos);

        const std::string_view key_name = header.substr(key_span.start, key_span.length);
        if (key_name == "__metadata__") {
            st_skip_value(header, pos);
        } else {
            SQUID_ASSERT(pos < header.size() && header[pos] == '{');
            ++pos;
            uint64_t start_off = 0;
            uint64_t end_off = 0;
            bool saw_offsets = false;

            for (uint32_t field_iter = 0; pos < header.size() && field_iter < 32; ++field_iter) {
                st_skip_ws(header, pos);
                if (pos < header.size() && header[pos] == '}') {
                    ++pos;
                    break;
                }
                const Span fk_span = st_parse_raw_string(header, pos);
                const std::string_view fk_name = header.substr(fk_span.start, fk_span.length);
                st_skip_ws(header, pos);
                SQUID_ASSERT(pos < header.size() && header[pos] == ':');
                ++pos;
                st_skip_ws(header, pos);

                if (fk_name == "data_offsets") {
                    SQUID_ASSERT(pos < header.size() && header[pos] == '[');
                    ++pos;
                    start_off = st_parse_u64(header, pos);
                    st_skip_ws(header, pos);
                    SQUID_ASSERT(pos < header.size() && header[pos] == ',');
                    ++pos;
                    end_off = st_parse_u64(header, pos);
                    st_skip_ws(header, pos);
                    SQUID_ASSERT(pos < header.size() && header[pos] == ']');
                    ++pos;
                    saw_offsets = true;
                } else {
                    st_skip_value(header, pos);
                }
                st_skip_ws(header, pos);
                if (pos < header.size() && header[pos] == ',') ++pos;
            }

            SQUID_ASSERT(saw_offsets);
            SQUID_ASSERT(end_off >= start_off);
            SQUID_ASSERT(data_base_offset + end_off <= file_bytes);

            TensorInfo info;
            info.name_offset = static_cast<uint32_t>(key_span.start);
            info.name_length = static_cast<uint16_t>(key_span.length);
            info.data_offset = start_off;
            info.byte_length = end_off - start_off;
            insert_tensor(header, info);
        }

        st_skip_ws(header, pos);
        if (pos < header.size() && header[pos] == ',') ++pos;
    }
}

} // namespace squid
