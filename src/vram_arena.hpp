#pragma once

// TigerStyle single-allocation VRAM bump arena & device memory guard (Spark unified memory or discrete VRAM).
// One contiguous cudaMalloc per lifetime, 256-byte alignment, exact Pass1==Pass2 accounting.

#include "config.hpp"
#include "tiger.hpp"
#include "../cuda/weights.hpp"
#include <cuda_runtime.h>
#include <fcntl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace squid {

constexpr uint64_t SPARK_MEMORY_LIMIT_BYTES = 96000000000ULL;
constexpr uint64_t SPARK_MEMORY_RESERVE_BYTES = 4ULL * 1024 * 1024 * 1024;

inline uint64_t linear_weight_vram_bytes(uint32_t rows, uint32_t cols, WeightKind kind) {
    SQUID_ASSERT(rows > 0 && cols > 0 && (cols % 16) == 0);
    switch (kind) {
        case WeightKind::BF16:
            return align_up_u64(static_cast<uint64_t>(rows) * cols * 2, VRAM_ALIGNMENT);
        case WeightKind::FP8:
            return align_up_u64(static_cast<uint64_t>(rows) * cols, VRAM_ALIGNMENT) +
                   align_up_u64(static_cast<uint64_t>(rows) * 4, VRAM_ALIGNMENT);
        case WeightKind::NVFP4:
            return align_up_u64(static_cast<uint64_t>(rows) * (cols / 2), VRAM_ALIGNMENT) +
                   align_up_u64(static_cast<uint64_t>(rows) * (cols / 16), VRAM_ALIGNMENT);
    }
    SQUID_ASSERT(false);
    return 0;
}

inline uint64_t kv_cache_pair_vram_bytes(uint32_t capacity_tokens, KVFormat fmt) {
    SQUID_ASSERT(capacity_tokens > 0);
    const uint64_t cap = capacity_tokens;
    switch (fmt) {
        case KVFormat::BF16:
            return 2 * align_up_u64(cap * KV_DIM * 2, VRAM_ALIGNMENT);
        case KVFormat::FP8:
            return 2 * align_up_u64(cap * KV_DIM, VRAM_ALIGNMENT) +
                   2 * align_up_u64(cap * NUM_KV_HEADS * 2, VRAM_ALIGNMENT);
        case KVFormat::Q4:
            return 2 * align_up_u64(cap * (KV_DIM / 2), VRAM_ALIGNMENT) +
                   2 * align_up_u64(cap * KV_Q4_SCALES_PER_TOK * 2, VRAM_ALIGNMENT);
    }
    SQUID_ASSERT(false);
    return 0;
}

inline bool read_spark_available_memory(uint64_t* out_bytes) {
    int fd = ::open("/proc/meminfo", O_RDONLY);
    if (fd < 0) return false;
    char buf[4096];
    ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
    ::close(fd);
    if (n <= 0) return false;
    buf[n] = '\0';

    const char* p = buf;
    while (*p) {
        const char* nl = std::strchr(p, '\n');
        size_t len = nl ? static_cast<size_t>(nl - p) : std::strlen(p);
        static constexpr char kPrefix[] = "MemAvailable:";
        if (len > sizeof(kPrefix) - 1 && std::memcmp(p, kPrefix, sizeof(kPrefix) - 1) == 0) {
            const char* q = p + sizeof(kPrefix) - 1;
            while (*q == ' ' || *q == '\t') ++q;
            char* endp = nullptr;
            unsigned long long kib = std::strtoull(q, &endp, 10);
            if (endp == q) return false;
            *out_bytes = static_cast<uint64_t>(kib) * 1024ULL;
            return true;
        }
        if (!nl) break;
        p = nl + 1;
    }
    return false;
}

inline bool check_spark_memory(uint64_t resident_bytes, uint64_t new_bytes) {
    const uint64_t needed_bytes = new_bytes + SPARK_MEMORY_RESERVE_BYTES;
    if (resident_bytes + needed_bytes > SPARK_MEMORY_LIMIT_BYTES) {
        fprintf(stderr, "[memory] Requested allocation exceeds the 96 GB budget with host/CUDA reserve\n");
        return false;
    }
    uint64_t avail_bytes = 0;
    if (!read_spark_available_memory(&avail_bytes)) {
        fprintf(stderr, "[memory] Could not read MemAvailable from /proc/meminfo\n");
        return false;
    }
    if (needed_bytes > avail_bytes) {
        fprintf(stderr, "[memory] Need %llu MiB of available Spark memory, found %llu MiB\n",
                static_cast<unsigned long long>(needed_bytes / (1024 * 1024)),
                static_cast<unsigned long long>(avail_bytes / (1024 * 1024)));
        return false;
    }
    return true;
}

// Discrete GPUs keep this much VRAM free for the CUDA context, cuBLAS, CUDA graphs and the
// vision encoder's per-request workspace. Override with SQUID_VRAM_RESERVE_MIB.
constexpr uint64_t DISCRETE_VRAM_RESERVE_BYTES = 1024ULL * 1024 * 1024;

inline uint64_t discrete_vram_reserve_bytes() {
    const char* env = std::getenv("SQUID_VRAM_RESERVE_MIB");
    if (env && env[0]) {
        char* endp = nullptr;
        const unsigned long long mib = std::strtoull(env, &endp, 10);
        if (endp != env) return static_cast<uint64_t>(mib) * 1024ULL * 1024ULL;
    }
    return DISCRETE_VRAM_RESERVE_BYTES;
}

// True when the GPU shares system DRAM with the CPU (DGX Spark GB10, Jetson).
inline bool device_is_integrated() {
    int dev = 0, integrated = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev) != cudaSuccess) return false;
    return integrated != 0;
}

// Bytes that can still be allocated for new_bytes-style requests after the reserve.
inline uint64_t device_allocatable_bytes(uint64_t resident_bytes) {
    if (device_is_integrated()) {
        uint64_t avail = 0;
        if (!read_spark_available_memory(&avail)) return 0;
        const uint64_t cap_left = (resident_bytes + SPARK_MEMORY_RESERVE_BYTES < SPARK_MEMORY_LIMIT_BYTES)
                                      ? SPARK_MEMORY_LIMIT_BYTES - resident_bytes - SPARK_MEMORY_RESERVE_BYTES : 0;
        const uint64_t avail_left = avail > SPARK_MEMORY_RESERVE_BYTES ? avail - SPARK_MEMORY_RESERVE_BYTES : 0;
        return cap_left < avail_left ? cap_left : avail_left;
    }
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    const uint64_t reserve = discrete_vram_reserve_bytes();
    return free_b > reserve ? static_cast<uint64_t>(free_b) - reserve : 0;
}

// resident_bytes: device memory this process already holds (only used for the Spark 96 GB cap;
// on discrete GPUs cudaMemGetInfo already reflects it).
inline bool check_device_memory(uint64_t resident_bytes, uint64_t new_bytes) {
    if (device_is_integrated()) return check_spark_memory(resident_bytes, new_bytes);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        fprintf(stderr, "[memory] cudaMemGetInfo failed\n");
        return false;
    }
    const uint64_t needed_bytes = new_bytes + discrete_vram_reserve_bytes();
    if (needed_bytes > free_b) {
        fprintf(stderr, "[memory] Need %llu MiB of free VRAM (incl. %llu MiB reserve), found %llu MiB of %llu MiB\n",
                static_cast<unsigned long long>(needed_bytes / (1024 * 1024)),
                static_cast<unsigned long long>(discrete_vram_reserve_bytes() / (1024 * 1024)),
                static_cast<unsigned long long>(free_b / (1024 * 1024)),
                static_cast<unsigned long long>(total_b / (1024 * 1024)));
        return false;
    }
    return true;
}

struct VRAMArena {
    uint8_t* base_ptr = nullptr;
    uint64_t capacity_bytes = 0;
    uint64_t offset_bytes = 0;
    uint32_t allocation_count = 0;

    void init(uint64_t cap_bytes) {
        SQUID_ASSERT(cap_bytes > 0);
        SQUID_ASSERT(base_ptr == nullptr);
        const uint64_t aligned_cap = align_up_u64(cap_bytes, VRAM_ALIGNMENT);
        void* raw = nullptr;
        CUDA_CHECK(cudaMalloc(&raw, static_cast<size_t>(aligned_cap)));
        SQUID_ASSERT(raw != nullptr);
        SQUID_ASSERT((reinterpret_cast<uintptr_t>(raw) % VRAM_ALIGNMENT) == 0);
        base_ptr = static_cast<uint8_t*>(raw);
        capacity_bytes = aligned_cap;
        offset_bytes = 0;
        allocation_count = 0;
    }

    void deinit() {
        SQUID_ASSERT(offset_bytes <= capacity_bytes);
        if (base_ptr) {
            CUDA_CHECK(cudaFree(base_ptr));
        }
        base_ptr = nullptr;
        capacity_bytes = 0;
        offset_bytes = 0;
        allocation_count = 0;
    }

    void* alloc_raw(uint64_t bytes) {
        SQUID_ASSERT(base_ptr != nullptr);
        SQUID_ASSERT(bytes > 0);
        const uint64_t aligned = align_up_u64(bytes, VRAM_ALIGNMENT);
        SQUID_ASSERT(aligned <= capacity_bytes - offset_bytes);
        uint8_t* ptr = base_ptr + offset_bytes;
        offset_bytes += aligned;
        allocation_count += 1;
        SQUID_ASSERT((reinterpret_cast<uintptr_t>(ptr) % VRAM_ALIGNMENT) == 0);
        SQUID_ASSERT(offset_bytes <= capacity_bytes);
        return ptr;
    }

    template <typename T>
    T* alloc_many(uint64_t count) {
        SQUID_ASSERT(count > 0);
        return static_cast<T*>(alloc_raw(count * sizeof(T)));
    }

    template <typename T>
    T* alloc_one() {
        return static_cast<T*>(alloc_raw(sizeof(T)));
    }

    void assert_exact_fill() const {
        SQUID_ASSERT(base_ptr != nullptr);
        SQUID_ASSERT(offset_bytes == capacity_bytes);
    }
};

} // namespace squid
