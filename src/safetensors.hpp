#pragma once

// TigerStyle POSIX mmap Safetensors loader (zero recursion, zero heap allocations).

#include "tiger.hpp"
#include <cstdint>
#include <string_view>

namespace squid {

constexpr uint32_t SAFETENSORS_MAX_TENSORS = 2048;
constexpr uint32_t SAFETENSORS_HASH_SLOTS = 4096;
constexpr uint64_t SAFETENSORS_HEADER_MAX = 16ULL * 1024 * 1024;

static_assert((SAFETENSORS_HASH_SLOTS & (SAFETENSORS_HASH_SLOTS - 1)) == 0);
static_assert(SAFETENSORS_HASH_SLOTS >= SAFETENSORS_MAX_TENSORS * 2);

struct TensorInfo {
    uint32_t name_offset = 0;
    uint16_t name_length = 0;
    uint64_t data_offset = 0;
    uint64_t byte_length = 0;
};

struct SafetensorsFile {
    static constexpr uint16_t kEmptySlot = 0xffffu;

    int file_descriptor = -1;
    const uint8_t* mapped_ptr = nullptr;
    uint64_t file_bytes = 0;
    uint64_t data_base_offset = 0;
    uint64_t header_bytes = 0;
    uint32_t tensor_count = 0;
    TensorInfo tensors[SAFETENSORS_MAX_TENSORS]{};
    uint16_t hash_table[SAFETENSORS_HASH_SLOTS]{};

    bool open_file(const char* path_z);
    void close_file();

    const TensorInfo* lookup_optional(std::string_view name) const;
    const TensorInfo* lookup(std::string_view name) const;

    const uint8_t* data_ptr(const TensorInfo* info) const {
        SQUID_ASSERT(mapped_ptr != nullptr && info != nullptr);
        const uint64_t start = data_base_offset + info->data_offset;
        const uint64_t end = start + info->byte_length;
        SQUID_ASSERT(end >= start && end <= file_bytes);
        return mapped_ptr + start;
    }

private:
    void insert_tensor(std::string_view header, const TensorInfo& info);
    void parse_header(std::string_view header);
};

} // namespace squid
