#include "weights.hpp"

#include "kernels.hpp"
#include "safetensors.hpp"
#include "tiger.hpp"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace squid {

namespace {

struct PinnedUploader {
    static constexpr size_t kChunkBytes = 16 * 1024 * 1024;
    uint8_t* bufs[2] = {nullptr, nullptr};
    cudaEvent_t events[2] = {nullptr, nullptr};
    bool in_flight[2] = {false, false};
    size_t slot = 0;

    void init() {
        for (int i = 0; i < 2; ++i) {
            void* p = nullptr;
            CUDA_CHECK(cudaHostAlloc(&p, kChunkBytes, cudaHostAllocDefault));
            bufs[i] = static_cast<uint8_t*>(p);
            CUDA_CHECK(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
            in_flight[i] = false;
        }
        slot = 0;
    }

    void deinit() {
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < 2; ++i) {
            if (events[i]) CUDA_CHECK(cudaEventDestroy(events[i]));
            if (bufs[i]) CUDA_CHECK(cudaFreeHost(bufs[i]));
            events[i] = nullptr;
            bufs[i] = nullptr;
        }
    }

    void upload(void* dst_dev, const uint8_t* src_host, size_t total_bytes, const SafetensorsFile* drop_sf) {
        auto* dst_u8 = static_cast<uint8_t*>(dst_dev);
        size_t off = 0;
        while (off < total_bytes) {
            const size_t n = std::min(kChunkBytes, total_bytes - off);
            const size_t s = slot;
            if (in_flight[s]) {
                CUDA_CHECK(cudaEventSynchronize(events[s]));
                in_flight[s] = false;
            }
            std::memcpy(bufs[s], src_host + off, n);
            if (drop_sf && n >= 2 * 1024 * 1024) {
                // Release private process PTEs to keep RSS low WITHOUT evicting OS page cache!
                const uintptr_t cur_addr = reinterpret_cast<uintptr_t>(src_host) + off;
                constexpr uintptr_t kPage = 4096;
                const uintptr_t s_addr = (cur_addr + kPage - 1) & ~(kPage - 1);
                const uintptr_t e_addr = (cur_addr + n) & ~(kPage - 1);
                if (e_addr > s_addr) {
                    (void)::madvise(reinterpret_cast<void*>(s_addr), static_cast<size_t>(e_addr - s_addr), MADV_DONTNEED);
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(dst_u8 + off, bufs[s], n, cudaMemcpyHostToDevice, nullptr));
            CUDA_CHECK(cudaEventRecord(events[s], nullptr));
            in_flight[s] = true;
            slot = s ^ 1;
            off += n;
        }
    }
};

void upload_raw(PinnedUploader& up, const SafetensorsFile& sf, std::string_view name, void* dst_dev, uint64_t expected_bytes) {
    const TensorInfo* info = sf.lookup(name);
    SQUID_ASSERT(info->byte_length == expected_bytes);
    const uint8_t* ptr = sf.data_ptr(info);
    up.upload(dst_dev, ptr, static_cast<size_t>(info->byte_length), &sf);
}

size_t stage_layer_raw(PinnedUploader& up, const SafetensorsFile& sf, uint32_t layer_idx, const char* suffix, uint8_t* dst_dev, uint64_t expected_bytes) {
    char name_buf[128];
    int n = std::snprintf(name_buf, sizeof(name_buf), "model.llm.layers.%u.%s", layer_idx, suffix);
    SQUID_ASSERT(n > 0 && static_cast<size_t>(n) < sizeof(name_buf));
    upload_raw(up, sf, std::string_view(name_buf, static_cast<size_t>(n)), dst_dev, expected_bytes);
    return static_cast<size_t>(expected_bytes);
}

void read_f32_or_bf16(const SafetensorsFile& sf, std::string_view name, float* dst, size_t count) {
    const TensorInfo* info = sf.lookup(name);
    const uint8_t* raw = sf.data_ptr(info);
    if (info->byte_length == count * 4) {
        std::memcpy(dst, raw, count * 4);
    } else if (info->byte_length == count * 2) {
        for (size_t i = 0; i < count; ++i) {
            uint16_t b16 = 0;
            std::memcpy(&b16, raw + i * 2, 2);
            uint32_t bits = static_cast<uint32_t>(b16) << 16;
            float val = 0.0f;
            std::memcpy(&val, &bits, 4);
            dst[i] = val;
        }
    } else {
        fprintf(stderr, "[ablation] Unexpected byte_length for %.*s: got %llu, expected %zu or %zu\n",
                static_cast<int>(name.size()), name.data(),
                static_cast<unsigned long long>(info->byte_length), count * 4, count * 2);
        std::abort();
    }
}

void normalize_rows_f32(float* buf, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        float* row = buf + r * cols;
        double ss = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            double d = row[c];
            ss += d * d;
        }
        const float norm = std::max(static_cast<float>(std::sqrt(ss)), 1e-12f);
        const float inv = 1.0f / norm;
        for (size_t c = 0; c < cols; ++c) row[c] *= inv;
    }
}

bool try_load_draft_vocab(std::string_view model_directory, QuantMode quant_mode, int32_t** out_ids, uint32_t* out_count) {
    *out_ids = nullptr;
    *out_count = VOCAB_UNPADDED;
    if (quant_mode == QuantMode::Exact) return false;
    const char* env_dv = std::getenv("SQUID_DRAFT_VOCAB");
    bool prefer_49k = false;
    if (env_dv) {
        std::string_view s(env_dv);
        if (s == "full") return false;
        if (s == "49k") prefer_49k = true;
    }
    const char* candidates[2] = {
        prefer_49k ? "/draft_vocab_49k.bin" : "/draft_vocab_32k.bin",
        prefer_49k ? "/draft_vocab_32k.bin" : "/draft_vocab_49k.bin",
    };

    char path_buf[1024];
    for (int ci = 0; ci < 2; ++ci) {
        const char* path_z = format_path_z(path_buf, sizeof(path_buf), model_directory, candidates[ci]);
        int fd = ::open(path_z, O_RDONLY);
        if (fd < 0) continue;
        off_t sz = ::lseek(fd, 0, SEEK_END);
        if (sz <= 0 || (sz % 4) != 0) {
            ::close(fd);
            continue;
        }
        (void)::lseek(fd, 0, SEEK_SET);
        const size_t count = static_cast<size_t>(sz / 4);
        if (count == 0 || count > static_cast<size_t>(VOCAB_UNPADDED)) {
            ::close(fd);
            continue;
        }
        int32_t* ids = alloc_host_zeroed<int32_t>(count);
        auto* raw_dst = reinterpret_cast<uint8_t*>(ids);
        const size_t want_bytes = count * 4;
        size_t got = 0;
        for (uint32_t iter = 0; got < want_bytes && iter < 4096; ++iter) {
            ssize_t n = ::read(fd, raw_dst + got, want_bytes - got);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            got += static_cast<size_t>(n);
        }
        ::close(fd);
        if (got == want_bytes) {
            bool all_valid = true;
            for (size_t i = 0; i < count; ++i) {
                if (ids[i] < 0 || ids[i] >= VOCAB_UNPADDED) {
                    all_valid = false;
                    break;
                }
            }
            if (all_valid) {
                *out_ids = ids;
                *out_count = static_cast<uint32_t>(count);
                return true;
            }
        }
        free_host(ids);
    }
    return false;
}

uint64_t compute_total_vram_bytes(
    QuantMode quant_mode,
    WeightKind dense_kind,
    WeightKind mtp_kind,
    bool has_mtp,
    bool has_pruned_draft_vocab,
    uint32_t draft_vocab_count,
    bool has_ablation,
    bool has_ablation_detectors) {
    constexpr uint64_t a = VRAM_ALIGNMENT;
    uint64_t total = 0;

    total += align_up_u64(static_cast<uint64_t>(VOCAB_PADDED) * HIDDEN * 2, a);
    total += align_up_u64(static_cast<uint64_t>(HIDDEN) * 2, a); // embed_norm
    total += align_up_u64(static_cast<uint64_t>(HIDDEN) * 2, a); // final_norm
    total += linear_weight_vram_bytes(VOCAB_UNPADDED, HIDDEN, dense_kind);

    for (int l = 1; l < NUM_LAYERS; ++l) {
        uint64_t elems = is_global_layer(l) ? (3ULL * HIDDEN) : static_cast<uint64_t>(HIDDEN);
        total += align_up_u64(elems * 4, a);
    }
    for (int l = 0; l < NUM_LAYERS; ++l) {
        total += align_up_u64(static_cast<uint64_t>(HIDDEN) * 4, a);
    }
    total += align_up_u64(static_cast<uint64_t>(HIDDEN) * 4, a);
    total += align_up_u64(24ULL * HIDDEN * 4, a);
    total += align_up_u64(24ULL * HIDDEN * 4, a);

    for (int l = 0; l < NUM_LAYERS; ++l) {
        const bool is_glob = is_global_layer(l);
        total += (is_glob ? 3 : 1) * align_up_u64(static_cast<uint64_t>(HIDDEN) * 2, a);
        total += align_up_u64(static_cast<uint64_t>(HIDDEN) * 2, a); // mlp_norm

        total += linear_weight_vram_bytes(QRGS_OUT_DIM, HIDDEN, dense_kind);
        total += linear_weight_vram_bytes(2 * KV_DIM, HIDDEN, dense_kind);
        total += linear_weight_vram_bytes(HIDDEN, Q_DIM, dense_kind);

        if (is_glob) {
            total += 2 * align_up_u64(static_cast<uint64_t>(CONV_TAPS) * KV_DIM * 2, a);
            total += 2 * align_up_u64(static_cast<uint64_t>(NUM_KV_HEADS) * CONV_TAPS * HIDDEN * 2, a);
        } else {
            total += 2 * align_up_u64(static_cast<uint64_t>(KV_DIM) * CONV_TAPS * 2, a);
        }
        total += 2 * align_up_u64(static_cast<uint64_t>(HIDDEN) * CONV_TAPS * 2, a);
        total += align_up_u64(static_cast<uint64_t>(NUM_HEADS) * HEAD_DIM * 2, a);
        total += align_up_u64(static_cast<uint64_t>(NUM_HEADS) * D_REL * 2, a);
        total += align_up_u64(static_cast<uint64_t>(D_REL) * REL_EXTENT * 2, a);

        if (l >= LOOP_START && l < LOOP_END) {
            for (int s = 0; s < 4; ++s) {
                if (quant_mode == QuantMode::NVFP4) {
                    total += linear_weight_vram_bytes(GATE_RANK, 2 * HIDDEN, WeightKind::NVFP4);
                    total += linear_weight_vram_bytes(HIDDEN, GATE_RANK, WeightKind::NVFP4);
                    total += linear_weight_vram_bytes(HIDDEN, GATE_RANK, WeightKind::NVFP4);
                } else {
                    total += align_up_u64(static_cast<uint64_t>(GATE_RANK) * 2 * HIDDEN * 2, a);
                    total += 2 * align_up_u64(static_cast<uint64_t>(HIDDEN) * GATE_RANK * 2, a);
                }
            }
        }

        if (quant_mode == QuantMode::NVFP4) {
            total += linear_weight_vram_bytes(NUM_EXPERTS, HIDDEN, WeightKind::NVFP4);
        } else {
            total += align_up_u64(static_cast<uint64_t>(NUM_EXPERTS) * HIDDEN * 2, a);
        }
        total += align_up_u64(static_cast<uint64_t>(NUM_EXPERTS) * 4, a);

        const uint64_t dims[2][2] = {
            {2ULL * INTERMEDIATE, static_cast<uint64_t>(HIDDEN)},
            {static_cast<uint64_t>(HIDDEN), static_cast<uint64_t>(INTERMEDIATE)},
        };
        const bool is_nvfp4_bank = is_nvfp4_expert_layer(l) || quant_mode == QuantMode::NVFP4;
        for (int di = 0; di < 2; ++di) {
            const uint64_t rows = dims[di][0];
            const uint64_t cols = dims[di][1];
            if (is_nvfp4_bank) {
                total += align_up_u64(static_cast<uint64_t>(NUM_EXPERTS) * rows * (cols / 2), a);
                total += align_up_u64(static_cast<uint64_t>(NUM_EXPERTS) * rows * (cols / 16), a);
                total += align_up_u64(static_cast<uint64_t>(NUM_EXPERTS) * 4, a);
            } else {
                total += align_up_u64(static_cast<uint64_t>(NUM_EXPERTS) * rows * cols, a);
                total += align_up_u64(static_cast<uint64_t>(NUM_EXPERTS) * rows * (cols / 32), a);
            }
        }
    }

    if (has_mtp) {
        total += align_up_u64(16 * 4, a);
        total += align_up_u64(static_cast<uint64_t>(HIDDEN) * 2, a);
        total += align_up_u64(static_cast<uint64_t>(HIDDEN) * 2, a);
        total += align_up_u64(static_cast<uint64_t>(HIDDEN) * 2, a);
        total += align_up_u64(static_cast<uint64_t>(VOCAB_PADDED) * MTP_MARKOV_RANK * 2, a);
        total += align_up_u64(static_cast<uint64_t>(3 * MTP_MARKOV_RANK) * 2, a);
        total += align_up_u64(768ULL * 512 * 2, a);
        total += 2 * align_up_u64(768ULL * HIDDEN * 2, a);

        if (has_pruned_draft_vocab) {
            total += align_up_u64(static_cast<uint64_t>(draft_vocab_count) * 4, a);
            total += align_up_u64(static_cast<uint64_t>(VOCAB_UNPADDED) * 4, a);
            total += linear_weight_vram_bytes(draft_vocab_count, HIDDEN, mtp_kind);
        }
        total += align_up_u64(static_cast<uint64_t>(draft_vocab_count) * 4, a);
        total += linear_weight_vram_bytes(draft_vocab_count, MTP_MARKOV_RANK, mtp_kind);

        for (int ml = 0; ml < MTP_LAYERS; ++ml) {
            total += 3 * align_up_u64(static_cast<uint64_t>(HIDDEN) * 2, a);
            total += 3 * align_up_u64(static_cast<uint64_t>(HEAD_DIM) * 2, a);
            total += linear_weight_vram_bytes(DSPARK_QKV_OUT_DIM, HIDDEN, mtp_kind);
            total += linear_weight_vram_bytes(HIDDEN, Q_DIM, mtp_kind);
            total += linear_weight_vram_bytes(2 * MTP_INTERMEDIATE, HIDDEN, mtp_kind);
            total += linear_weight_vram_bytes(HIDDEN, MTP_INTERMEDIATE, mtp_kind);
            total += linear_weight_vram_bytes(2 * KV_DIM, HIDDEN, mtp_kind);
        }
    }

    if (has_ablation) {
        const uint64_t dir_bytes = static_cast<uint64_t>(NUM_VISITS) * HIDDEN * 4;
        const uint64_t param_bytes = static_cast<uint64_t>(NUM_VISITS) * 4 * 4;
        total += 2 * align_up_u64(dir_bytes, a);
        total += 2 * align_up_u64(param_bytes, a);
        if (has_ablation_detectors) {
            total += 2 * align_up_u64(dir_bytes, a);
        }
    }

    return total;
}

__nv_bfloat16* upload_bf16(VRAMArena& arena, PinnedUploader& up, const SafetensorsFile& sf, std::string_view name, uint64_t elems) {
    const uint64_t bytes = elems * 2;
    void* dst = arena.alloc_raw(bytes);
    upload_raw(up, sf, name, dst, bytes);
    return static_cast<__nv_bfloat16*>(dst);
}

__nv_bfloat16* upload_layer_bf16(VRAMArena& arena, PinnedUploader& up, const SafetensorsFile& sf, uint32_t layer_idx, const char* suffix, uint64_t elems) {
    char name_buf[128];
    int n = std::snprintf(name_buf, sizeof(name_buf), "model.llm.layers.%u.%s", layer_idx, suffix);
    SQUID_ASSERT(n > 0 && static_cast<size_t>(n) < sizeof(name_buf));
    return upload_bf16(arena, up, sf, std::string_view(name_buf, static_cast<size_t>(n)), elems);
}

float* upload_f32(VRAMArena& arena, PinnedUploader& up, const SafetensorsFile& sf, std::string_view name, uint64_t elems) {
    const uint64_t bytes = elems * 4;
    float* dst = arena.alloc_many<float>(elems);
    upload_raw(up, sf, name, dst, bytes);
    return dst;
}

LinearWeight pack_linear(VRAMArena& arena, const void* src_bf16_dev, uint32_t rows, uint32_t cols, WeightKind kind, float* d_amax) {
    SQUID_ASSERT(rows > 0 && cols > 0 && (cols % 16) == 0);
    LinearWeight lw{};
    lw.kind = kind;
    lw.rows = static_cast<int>(rows);
    lw.cols = static_cast<int>(cols);
    lw.global_scale = 1.0f;

    switch (kind) {
        case WeightKind::BF16: {
            const uint64_t bytes = static_cast<uint64_t>(rows) * cols * 2;
            lw.data = arena.alloc_raw(bytes);
            CUDA_CHECK(cudaMemcpyAsync(lw.data, src_bf16_dev, static_cast<size_t>(bytes), cudaMemcpyDeviceToDevice, nullptr));
            break;
        }
        case WeightKind::FP8: {
            lw.data = arena.alloc_raw(static_cast<uint64_t>(rows) * cols);
            lw.row_scales = arena.alloc_many<float>(rows);
            pack_bf16_to_fp8(src_bf16_dev, lw.data, lw.row_scales, static_cast<int32_t>(rows), static_cast<int32_t>(cols));
            break;
        }
        case WeightKind::NVFP4: {
            lw.data = arena.alloc_raw(static_cast<uint64_t>(rows) * (cols / 2));
            lw.scales = arena.alloc_many<uint8_t>(static_cast<uint64_t>(rows) * (cols / 16));
            lw.global_scale = pack_bf16_to_nvfp4(src_bf16_dev, lw.data, lw.scales, d_amax,
                                                static_cast<int32_t>(rows), static_cast<int32_t>(cols));
            break;
        }
    }
    return lw;
}

LinearWeight pack_linear_chunked(
    VRAMArena& arena,
    PinnedUploader& up,
    const uint8_t* src_host_bf16,
    size_t src_bytes,
    uint32_t rows,
    uint32_t cols,
    WeightKind kind,
    void* d_stage,
    float* d_amax,
    const SafetensorsFile* drop_sf) {
    SQUID_ASSERT(rows > 0 && cols > 0 && (cols % 16) == 0);
    const size_t row_bytes = static_cast<size_t>(cols) * 2;
    SQUID_ASSERT(src_bytes == static_cast<size_t>(rows) * row_bytes);

    LinearWeight lw{};
    lw.kind = kind;
    lw.rows = static_cast<int>(rows);
    lw.cols = static_cast<int>(cols);
    lw.global_scale = 1.0f;

    constexpr uint32_t kChunkRows = 16384; // 16384 * 2304 * 2 = 75.5 MiB < 96 MiB d_stage
    switch (kind) {
        case WeightKind::BF16: {
            lw.data = arena.alloc_raw(src_bytes);
            up.upload(lw.data, src_host_bf16, src_bytes, drop_sf);
            break;
        }
        case WeightKind::FP8: {
            auto* dst_w = arena.alloc_many<uint8_t>(static_cast<uint64_t>(rows) * cols);
            auto* dst_rs = arena.alloc_many<float>(rows);
            lw.data = dst_w;
            lw.row_scales = dst_rs;
            for (uint32_t r0 = 0; r0 < rows; ) {
                const uint32_t rc = std::min(kChunkRows, rows - r0);
                up.upload(d_stage, src_host_bf16 + static_cast<size_t>(r0) * row_bytes, static_cast<size_t>(rc) * row_bytes, drop_sf);
                pack_bf16_to_fp8(d_stage, dst_w + static_cast<size_t>(r0) * cols, dst_rs + r0,
                                 static_cast<int32_t>(rc), static_cast<int32_t>(cols));
                r0 += rc;
            }
            break;
        }
        case WeightKind::NVFP4: {
            auto* dst_w = arena.alloc_many<uint8_t>(static_cast<uint64_t>(rows) * (cols / 2));
            auto* dst_s = arena.alloc_many<uint8_t>(static_cast<uint64_t>(rows) * (cols / 16));
            lw.data = dst_w;
            lw.scales = dst_s;
            CUDA_CHECK(cudaMemsetAsync(d_amax, 0, sizeof(float), nullptr));
            for (uint32_t r0 = 0; r0 < rows; ) {
                const uint32_t rc = std::min(kChunkRows, rows - r0);
                up.upload(d_stage, src_host_bf16 + static_cast<size_t>(r0) * row_bytes, static_cast<size_t>(rc) * row_bytes, drop_sf);
                launch_bf16_amax_accum(d_stage, static_cast<size_t>(rc) * cols, d_amax, nullptr);
                r0 += rc;
            }
            float h_amax = 0.0f;
            CUDA_CHECK(cudaMemcpy(&h_amax, d_amax, sizeof(float), cudaMemcpyDeviceToHost));
            const float gscale = std::max(h_amax / (448.0f * 6.0f), 1e-12f);
            lw.global_scale = gscale;
            for (uint32_t r0 = 0; r0 < rows; ) {
                const uint32_t rc = std::min(kChunkRows, rows - r0);
                up.upload(d_stage, src_host_bf16 + static_cast<size_t>(r0) * row_bytes, static_cast<size_t>(rc) * row_bytes, drop_sf);
                launch_pack_bf16_to_nvfp4_with_scale(
                    d_stage,
                    dst_w + static_cast<size_t>(r0) * (cols / 2),
                    dst_s + static_cast<size_t>(r0) * (cols / 16),
                    gscale,
                    static_cast<int32_t>(rc),
                    static_cast<int32_t>(cols),
                    nullptr);
                r0 += rc;
            }
            break;
        }
    }
    return lw;
}

ExpertBank load_expert_bank(
    VRAMArena& arena,
    PinnedUploader& up,
    const SafetensorsFile& sf,
    uint32_t layer_idx,
    const char* wname,
    uint32_t rows,
    uint32_t cols,
    QuantMode quant_mode,
    void* d_stage) {
    ExpertBank eb{};
    eb.n_experts = NUM_EXPERTS;
    eb.rows = static_cast<int>(rows);
    eb.cols = static_cast<int>(cols);

    char nbuf[128];
    const bool ckpt_nvfp4 = is_nvfp4_expert_layer(static_cast<int>(layer_idx));
    if (ckpt_nvfp4) {
        eb.is_nvfp4 = true;
        const uint64_t w_bytes = static_cast<uint64_t>(NUM_EXPERTS) * rows * (cols / 2);
        const uint64_t s_bytes = static_cast<uint64_t>(NUM_EXPERTS) * rows * (cols / 16);
        eb.data = arena.alloc_many<uint8_t>(w_bytes);
        eb.scales = arena.alloc_many<uint8_t>(s_bytes);

        int n = std::snprintf(nbuf, sizeof(nbuf), "model.llm.layers.%u.mlp.experts.%s_weight", layer_idx, wname);
        upload_raw(up, sf, std::string_view(nbuf, static_cast<size_t>(n)), eb.data, w_bytes);
        n = std::snprintf(nbuf, sizeof(nbuf), "model.llm.layers.%u.mlp.experts.%s_weight_scale", layer_idx, wname);
        upload_raw(up, sf, std::string_view(nbuf, static_cast<size_t>(n)), eb.scales, s_bytes);

        n = std::snprintf(nbuf, sizeof(nbuf), "model.llm.layers.%u.mlp.experts.%s_weight_amax", layer_idx, wname);
        const TensorInfo* ainfo = sf.lookup(std::string_view(nbuf, static_cast<size_t>(n)));
        const uint8_t* amax_raw = sf.data_ptr(ainfo);
        n = std::snprintf(nbuf, sizeof(nbuf), "model.llm.layers.%u.mlp.experts.%s_weight_outer_scale_denominator", layer_idx, wname);
        const TensorInfo* dinfo = sf.lookup(std::string_view(nbuf, static_cast<size_t>(n)));
        const uint8_t* denom_raw = sf.data_ptr(dinfo);
        SQUID_ASSERT(ainfo->byte_length == static_cast<uint64_t>(NUM_EXPERTS) * 4);
        SQUID_ASSERT(dinfo->byte_length >= 4);

        float denom = 1.0f;
        std::memcpy(&denom, denom_raw, 4);
        const float inv_denom = 1.0f / denom;
        float h_scales[NUM_EXPERTS];
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            float amax_val = 0.0f;
            std::memcpy(&amax_val, amax_raw + e * 4, 4);
            h_scales[e] = amax_val * inv_denom;
        }
        eb.expert_scale = arena.alloc_many<float>(NUM_EXPERTS);
        up.upload(eb.expert_scale, reinterpret_cast<const uint8_t*>(h_scales), sizeof(h_scales), nullptr);
    } else if (quant_mode != QuantMode::NVFP4) {
        eb.is_nvfp4 = false;
        const uint64_t w_bytes = static_cast<uint64_t>(NUM_EXPERTS) * rows * cols;
        const uint64_t s_bytes = static_cast<uint64_t>(NUM_EXPERTS) * rows * (cols / 32);
        eb.data = arena.alloc_many<uint8_t>(w_bytes);
        eb.scales = arena.alloc_many<uint8_t>(s_bytes);
        int n = std::snprintf(nbuf, sizeof(nbuf), "model.llm.layers.%u.mlp.experts.%s_weight", layer_idx, wname);
        upload_raw(up, sf, std::string_view(nbuf, static_cast<size_t>(n)), eb.data, w_bytes);
        n = std::snprintf(nbuf, sizeof(nbuf), "model.llm.layers.%u.mlp.experts.%s_weight_scale", layer_idx, wname);
        upload_raw(up, sf, std::string_view(nbuf, static_cast<size_t>(n)), eb.scales, s_bytes);
    } else {
        eb.is_nvfp4 = true;
        const size_t per_exp_mx_w = static_cast<size_t>(rows) * cols;
        const size_t per_exp_mx_s = static_cast<size_t>(rows) * (cols / 32);
        const size_t per_exp_nv_w = static_cast<size_t>(rows) * (cols / 2);
        const size_t per_exp_nv_s = static_cast<size_t>(rows) * (cols / 16);

        int n = std::snprintf(nbuf, sizeof(nbuf), "model.llm.layers.%u.mlp.experts.%s_weight", layer_idx, wname);
        const TensorInfo* w_info = sf.lookup(std::string_view(nbuf, static_cast<size_t>(n)));
        SQUID_ASSERT(w_info->byte_length == static_cast<uint64_t>(NUM_EXPERTS) * per_exp_mx_w);
        const uint8_t* w_ptr = sf.data_ptr(w_info);

        n = std::snprintf(nbuf, sizeof(nbuf), "model.llm.layers.%u.mlp.experts.%s_weight_scale", layer_idx, wname);
        const TensorInfo* s_info = sf.lookup(std::string_view(nbuf, static_cast<size_t>(n)));
        SQUID_ASSERT(s_info->byte_length == static_cast<uint64_t>(NUM_EXPERTS) * per_exp_mx_s);
        const uint8_t* s_ptr = sf.data_ptr(s_info);

        eb.expert_scale = arena.alloc_many<float>(NUM_EXPERTS);
        eb.data = arena.alloc_many<uint8_t>(static_cast<uint64_t>(NUM_EXPERTS) * per_exp_nv_w);
        eb.scales = arena.alloc_many<uint8_t>(static_cast<uint64_t>(NUM_EXPERTS) * per_exp_nv_s);

        constexpr uint32_t kExpChunk = 16;
        auto* tmp_w = static_cast<uint8_t*>(d_stage);
        uint8_t* tmp_s = tmp_w + static_cast<size_t>(kExpChunk) * per_exp_mx_w;
        for (uint32_t e0 = 0; e0 < static_cast<uint32_t>(NUM_EXPERTS); e0 += kExpChunk) {
            const uint32_t ec = std::min(kExpChunk, static_cast<uint32_t>(NUM_EXPERTS) - e0);
            up.upload(tmp_w, w_ptr + static_cast<size_t>(e0) * per_exp_mx_w, static_cast<size_t>(ec) * per_exp_mx_w, &sf);
            up.upload(tmp_s, s_ptr + static_cast<size_t>(e0) * per_exp_mx_s, static_cast<size_t>(ec) * per_exp_mx_s, &sf);
            convert_mxfp8_to_nvfp4_experts(
                tmp_w, tmp_s,
                eb.data + static_cast<size_t>(e0) * per_exp_nv_w,
                eb.scales + static_cast<size_t>(e0) * per_exp_nv_s,
                eb.expert_scale + e0,
                static_cast<int32_t>(ec), static_cast<int32_t>(rows), static_cast<int32_t>(cols));
        }
    }
    return eb;
}

} // namespace

void LoadedModel::deinit() {
    arena.deinit();
    mw = ModelWeights{};
    ablation = AblationWeights{};
}

bool LoadedModel::load(
    std::string_view model_directory,
    QuantMode quant_mode,
    bool load_mtp,
    std::string_view ablation_path) {
    SQUID_ASSERT(!model_directory.empty());
    SQUID_ASSERT(arena.base_ptr == nullptr);

    const uint64_t start_ns = clock_monotonic_ns();
    mw = ModelWeights{};
    ablation = AblationWeights{};
    mw.mode = quant_mode;

    char path_buf[1024];
    SafetensorsFile sf_main{};
    if (!sf_main.open_file(format_path_z(path_buf, sizeof(path_buf), model_directory, "/model-00001-of-00001.safetensors"))) {
        fprintf(stderr, "[weights] Failed to open %.*s/model-00001-of-00001.safetensors\n",
                static_cast<int>(model_directory.size()), model_directory.data());
        return false;
    }

    SafetensorsFile sf_mtp{};
    const bool has_mtp = load_mtp && sf_mtp.open_file(format_path_z(path_buf, sizeof(path_buf), model_directory, "/mtp.safetensors"));

    SafetensorsFile sf_abl{};
    bool has_ablation = false;
    bool has_ablation_detectors = false;
    if (!ablation_path.empty()) {
        SQUID_ASSERT(ablation_path.size() < sizeof(path_buf));
        std::memcpy(path_buf, ablation_path.data(), ablation_path.size());
        path_buf[ablation_path.size()] = '\0';
        if (!sf_abl.open_file(path_buf)) {
            fprintf(stderr, "[weights] Failed to open ablation safetensors: %.*s\n",
                    static_cast<int>(ablation_path.size()), ablation_path.data());
            if (has_mtp) sf_mtp.close_file();
            sf_main.close_file();
            return false;
        }
        has_ablation = true;
        const bool plain_layout = sf_abl.lookup_optional("attn_w") != nullptr;
        has_ablation_detectors = !plain_layout && sf_abl.lookup_optional("attn_det") != nullptr;
    }

    const WeightKind dense_kind = (quant_mode == QuantMode::Exact) ? WeightKind::BF16
                                : (quant_mode == QuantMode::FP8)   ? WeightKind::FP8
                                                                   : WeightKind::NVFP4;
    const char* env_mtp_bf16 = std::getenv("SQUID_MTP_BF16");
    const bool force_mtp_bf16 = (env_mtp_bf16 != nullptr && env_mtp_bf16[0] == '1');
    const WeightKind mtp_kind = (quant_mode == QuantMode::Exact || force_mtp_bf16) ? WeightKind::BF16 : WeightKind::NVFP4;

    int32_t* draft_vocab_host = nullptr;
    uint32_t draft_vocab_count = VOCAB_UNPADDED;
    bool has_pruned_draft_vocab = false;
    if (has_mtp) {
        has_pruned_draft_vocab = try_load_draft_vocab(model_directory, quant_mode, &draft_vocab_host, &draft_vocab_count);
    }

    const uint64_t total_vram_bytes = compute_total_vram_bytes(
        quant_mode, dense_kind, mtp_kind, has_mtp,
        has_pruned_draft_vocab, draft_vocab_count, has_ablation, has_ablation_detectors);
    constexpr uint64_t kStageBytes = 96ULL * 1024 * 1024;
    const uint64_t fold_temp_bytes = has_mtp ? (static_cast<uint64_t>(draft_vocab_count) * MTP_MARKOV_RANK * 2 + 2ULL * 1024 * 1024) : 0;
    if (!check_spark_memory(0, total_vram_bytes + kStageBytes + fold_temp_bytes)) {
        if (draft_vocab_host) free_host(draft_vocab_host);
        if (has_ablation) sf_abl.close_file();
        if (has_mtp) sf_mtp.close_file();
        sf_main.close_file();
        return false;
    }

    arena.init(total_vram_bytes);

    PinnedUploader up{};
    up.init();

    void* d_stage = nullptr;
    CUDA_CHECK(cudaMalloc(&d_stage, static_cast<size_t>(kStageBytes)));
    float* d_amax = nullptr;
    CUDA_CHECK(cudaMalloc(&d_amax, sizeof(float)));

    // --- PASS 2: Upload & Pack Directly Into Contiguous VRAMArena ---
    mw.embed = upload_bf16(arena, up, sf_main, "model.llm.embed.weight", static_cast<uint64_t>(VOCAB_PADDED) * HIDDEN);
    mw.embed_norm = upload_bf16(arena, up, sf_main, "model.llm.embed_norm.weight", HIDDEN);
    mw.final_norm = upload_bf16(arena, up, sf_main, "model.llm.norm.weight", HIDDEN);

    const bool fuse_unembed_with_dspark_fold = (has_mtp && !has_pruned_draft_vocab && dense_kind == WeightKind::NVFP4);
    if (!fuse_unembed_with_dspark_fold) {
        const TensorInfo* ui = sf_main.lookup("model.llm.unembed.weight");
        const size_t usable_bytes = static_cast<size_t>(VOCAB_UNPADDED) * HIDDEN * 2;
        SQUID_ASSERT(ui->byte_length >= usable_bytes);
        mw.unembed = pack_linear_chunked(
            arena, up, sf_main.data_ptr(ui), usable_bytes,
            VOCAB_UNPADDED, HIDDEN, dense_kind, d_stage, d_amax, &sf_main);
    }

    char name_buf[128];
    for (int l = 1; l < NUM_LAYERS; ++l) {
        uint64_t elems = is_global_layer(l) ? (3ULL * HIDDEN) : static_cast<uint64_t>(HIDDEN);
        int n = std::snprintf(name_buf, sizeof(name_buf), "model.llm.attn_res.attn_queries.%d", l - 1);
        mw.attn_queries[l - 1] = upload_f32(arena, up, sf_main, std::string_view(name_buf, static_cast<size_t>(n)), elems);
    }
    for (int l = 0; l < NUM_LAYERS; ++l) {
        int n = std::snprintf(name_buf, sizeof(name_buf), "model.llm.attn_res.mlp_queries.%d", l);
        mw.mlp_queries[l] = upload_f32(arena, up, sf_main, std::string_view(name_buf, static_cast<size_t>(n)), HIDDEN);
    }
    mw.output_query = upload_f32(arena, up, sf_main, "model.llm.attn_res.output_query", HIDDEN);
    mw.cross_pass_queries = upload_f32(arena, up, sf_main, "model.llm.middle_cycle_cross_pass_queries", 24ULL * HIDDEN);
    mw.cross_pass_kv_queries = upload_f32(arena, up, sf_main, "model.llm.middle_cycle_cross_pass_kv_queries", 24ULL * HIDDEN);

    for (uint32_t l = 0; l < static_cast<uint32_t>(NUM_LAYERS); ++l) {
        LayerWeights& lw = mw.layers[l];
        lw.is_global = is_global_layer(static_cast<int>(l));
        if (lw.is_global) {
            lw.q_in_norm = upload_layer_bf16(arena, up, sf_main, l, "q_input_norm.weight", HIDDEN);
            lw.k_in_norm = upload_layer_bf16(arena, up, sf_main, l, "k_input_norm.weight", HIDDEN);
            lw.v_in_norm = upload_layer_bf16(arena, up, sf_main, l, "v_input_norm.weight", HIDDEN);
        } else {
            lw.attn_norm = upload_layer_bf16(arena, up, sf_main, l, "attn_norm.weight", HIDDEN);
        }
        lw.mlp_norm = upload_layer_bf16(arena, up, sf_main, l, "mlp_norm.weight", HIDDEN);

        auto* stage_u8 = static_cast<uint8_t*>(d_stage);
        {
            size_t off = 0;
            off += stage_layer_raw(up, sf_main, l, "attn.wq_du.weight", stage_u8 + off, static_cast<uint64_t>(Q_DIM) * HIDDEN * 2);
            off += stage_layer_raw(up, sf_main, l, "attn.wr_du.weight", stage_u8 + off, static_cast<uint64_t>(R_DIM) * HIDDEN * 2);
            off += stage_layer_raw(up, sf_main, l, "attn.gate_attn.weight", stage_u8 + off, static_cast<uint64_t>(Q_DIM) * HIDDEN * 2);
            off += stage_layer_raw(up, sf_main, l, "attn.attn_sink_proj.weight", stage_u8 + off, static_cast<uint64_t>(NUM_HEADS) * HIDDEN * 2);
            SQUID_ASSERT(off == static_cast<size_t>(QRGS_OUT_DIM) * HIDDEN * 2);
            lw.w_qrgs = pack_linear(arena, d_stage, QRGS_OUT_DIM, HIDDEN, dense_kind, d_amax);
        }
        {
            const uint64_t kv_bytes = static_cast<uint64_t>(KV_DIM) * HIDDEN * 2;
            (void)stage_layer_raw(up, sf_main, l, "attn.wk_dv.weight", stage_u8, kv_bytes);
            (void)stage_layer_raw(up, sf_main, l, "attn.wv_dv.weight", stage_u8 + kv_bytes, kv_bytes);
            lw.w_kv = pack_linear(arena, d_stage, 2 * KV_DIM, HIDDEN, dense_kind, d_amax);

            lw.w_k = lw.w_kv;
            lw.w_k.rows = KV_DIM;
            lw.w_v = lw.w_kv;
            lw.w_v.rows = KV_DIM;
            auto* base_data = static_cast<uint8_t*>(lw.w_kv.data);
            switch (dense_kind) {
                case WeightKind::BF16:
                    lw.w_v.data = base_data + static_cast<size_t>(KV_DIM) * HIDDEN * 2;
                    break;
                case WeightKind::FP8:
                    lw.w_v.data = base_data + static_cast<size_t>(KV_DIM) * HIDDEN;
                    lw.w_v.row_scales = lw.w_kv.row_scales + KV_DIM;
                    break;
                case WeightKind::NVFP4:
                    lw.w_v.data = base_data + static_cast<size_t>(KV_DIM) * (HIDDEN / 2);
                    lw.w_v.scales = lw.w_kv.scales + static_cast<size_t>(KV_DIM) * (HIDDEN / 16);
                    break;
            }
        }
        {
            (void)stage_layer_raw(up, sf_main, l, "attn.wo_ud.weight", stage_u8, static_cast<uint64_t>(HIDDEN) * Q_DIM * 2);
            lw.w_o = pack_linear(arena, d_stage, HIDDEN, Q_DIM, dense_kind, d_amax);
        }

        if (lw.is_global) {
            lw.k_sconv_w = upload_layer_bf16(arena, up, sf_main, l, "attn.k_sconv.static_w", CONV_TAPS * KV_DIM);
            lw.v_sconv_w = upload_layer_bf16(arena, up, sf_main, l, "attn.v_sconv.static_w", CONV_TAPS * KV_DIM);
            lw.k_sconv_proj = upload_layer_bf16(arena, up, sf_main, l, "attn.k_sconv.weight_proj", static_cast<uint64_t>(NUM_KV_HEADS) * CONV_TAPS * HIDDEN);
            lw.v_sconv_proj = upload_layer_bf16(arena, up, sf_main, l, "attn.v_sconv.weight_proj", static_cast<uint64_t>(NUM_KV_HEADS) * CONV_TAPS * HIDDEN);
        } else {
            lw.k_sconv_w = upload_layer_bf16(arena, up, sf_main, l, "attn.k_sconv.weight", KV_DIM * CONV_TAPS);
            lw.v_sconv_w = upload_layer_bf16(arena, up, sf_main, l, "attn.v_sconv.weight", KV_DIM * CONV_TAPS);
        }
        lw.attn_sconv_w = upload_layer_bf16(arena, up, sf_main, l, "attn_sconv.weight", HIDDEN * CONV_TAPS);
        lw.mlp_sconv_w = upload_layer_bf16(arena, up, sf_main, l, "mlp_sconv.weight", HIDDEN * CONV_TAPS);
        lw.q_clip_scale = upload_layer_bf16(arena, up, sf_main, l, "attn.q_l2_clip.scale", NUM_HEADS * HEAD_DIM);
        lw.rel_clip_scale = upload_layer_bf16(arena, up, sf_main, l, "attn.rel_l2_clip.scale", NUM_HEADS * D_REL);
        lw.rel_logits_proj = upload_layer_bf16(arena, up, sf_main, l, "attn.rel_logits_proj.proj", D_REL * REL_EXTENT);

        if (l >= static_cast<uint32_t>(LOOP_START) && l < static_cast<uint32_t>(LOOP_END)) {
            lw.has_controllers = true;
            static constexpr const char* kStems[4] = {"attn", "k", "v", "mlp"};
            for (int s = 0; s < 4; ++s) {
                if (quant_mode == QuantMode::NVFP4) {
                    int n = std::snprintf(name_buf, sizeof(name_buf), "model.llm.layers.%u.controllers.%s.down_weight", l, kStems[s]);
                    upload_raw(up, sf_main, std::string_view(name_buf, static_cast<size_t>(n)), d_stage, static_cast<uint64_t>(GATE_RANK) * 2 * HIDDEN * 2);
                    lw.ctrl[s].down_nvfp4 = pack_linear(arena, d_stage, GATE_RANK, 2 * HIDDEN, WeightKind::NVFP4, d_amax);

                    n = std::snprintf(name_buf, sizeof(name_buf), "model.llm.layers.%u.controllers.%s.forget_weight", l, kStems[s]);
                    upload_raw(up, sf_main, std::string_view(name_buf, static_cast<size_t>(n)), d_stage, static_cast<uint64_t>(HIDDEN) * GATE_RANK * 2);
                    lw.ctrl[s].forget_nvfp4 = pack_linear(arena, d_stage, HIDDEN, GATE_RANK, WeightKind::NVFP4, d_amax);

                    n = std::snprintf(name_buf, sizeof(name_buf), "model.llm.layers.%u.controllers.%s.gain_weight", l, kStems[s]);
                    upload_raw(up, sf_main, std::string_view(name_buf, static_cast<size_t>(n)), d_stage, static_cast<uint64_t>(HIDDEN) * GATE_RANK * 2);
                    lw.ctrl[s].gain_nvfp4 = pack_linear(arena, d_stage, HIDDEN, GATE_RANK, WeightKind::NVFP4, d_amax);
                } else {
                    int n = std::snprintf(name_buf, sizeof(name_buf), "model.llm.layers.%u.controllers.%s.down_weight", l, kStems[s]);
                    lw.ctrl[s].down_w = upload_bf16(arena, up, sf_main, std::string_view(name_buf, static_cast<size_t>(n)), static_cast<uint64_t>(GATE_RANK) * 2 * HIDDEN);
                    n = std::snprintf(name_buf, sizeof(name_buf), "model.llm.layers.%u.controllers.%s.forget_weight", l, kStems[s]);
                    lw.ctrl[s].forget_w = upload_bf16(arena, up, sf_main, std::string_view(name_buf, static_cast<size_t>(n)), static_cast<uint64_t>(HIDDEN) * GATE_RANK);
                    n = std::snprintf(name_buf, sizeof(name_buf), "model.llm.layers.%u.controllers.%s.gain_weight", l, kStems[s]);
                    lw.ctrl[s].gain_w = upload_bf16(arena, up, sf_main, std::string_view(name_buf, static_cast<size_t>(n)), static_cast<uint64_t>(HIDDEN) * GATE_RANK);
                }
            }
        }

        if (quant_mode == QuantMode::NVFP4) {
            (void)stage_layer_raw(up, sf_main, l, "mlp.gate.weight", stage_u8, static_cast<uint64_t>(NUM_EXPERTS) * HIDDEN * 2);
            lw.router_nvfp4 = pack_linear(arena, d_stage, NUM_EXPERTS, HIDDEN, WeightKind::NVFP4, d_amax);
        } else {
            lw.router_w = upload_layer_bf16(arena, up, sf_main, l, "mlp.gate.weight", static_cast<uint64_t>(NUM_EXPERTS) * HIDDEN);
        }
        int n = std::snprintf(name_buf, sizeof(name_buf), "model.llm.layers.%u.mlp.gate.bias", l);
        lw.router_bias = upload_f32(arena, up, sf_main, std::string_view(name_buf, static_cast<size_t>(n)), NUM_EXPERTS);

        lw.w13 = load_expert_bank(arena, up, sf_main, l, "w13", 2 * INTERMEDIATE, HIDDEN, quant_mode, d_stage);
        lw.w2 = load_expert_bank(arena, up, sf_main, l, "w2", HIDDEN, INTERMEDIATE, quant_mode, d_stage);
    }

    // --- DSpark MTP Weights ---
    if (has_mtp) {
        const TensorInfo* fwi = sf_mtp.lookup("fuse_weight");
        SQUID_ASSERT(fwi->byte_length == 16 * 2);
        const uint8_t* fw_ptr = sf_mtp.data_ptr(fwi);
        float alpha[16];
        for (int row = 0; row < 4; ++row) {
            double vals[4];
            double max_val = -1e30;
            for (int col = 0; col < 4; ++col) {
                uint16_t b16 = 0;
                std::memcpy(&b16, fw_ptr + (row * 4 + col) * 2, 2);
                uint32_t bits = static_cast<uint32_t>(b16) << 16;
                float vf = 0.0f;
                std::memcpy(&vf, &bits, 4);
                vals[col] = vf;
                if (vals[col] > max_val) max_val = vals[col];
            }
            double sum_exp = 0.0;
            for (int col = 0; col < 4; ++col) {
                vals[col] = std::exp(vals[col] - max_val);
                sum_exp += vals[col];
            }
            for (int col = 0; col < 4; ++col) {
                alpha[row * 4 + col] = static_cast<float>(vals[col] / sum_exp);
            }
        }
        mw.dspark.fuse_alpha = arena.alloc_many<float>(16);
        up.upload(mw.dspark.fuse_alpha, reinterpret_cast<const uint8_t*>(alpha), sizeof(alpha), nullptr);

        mw.dspark.mask_embed = upload_bf16(arena, up, sf_mtp, "mask_embed", HIDDEN);
        mw.dspark.norm_w = upload_bf16(arena, up, sf_mtp, "norm.weight", HIDDEN);
        mw.dspark.markov_dflare_norm = upload_bf16(arena, up, sf_mtp, "markov_head.dflare_norm.weight", HIDDEN);
        mw.dspark.markov_w1 = upload_bf16(arena, up, sf_mtp, "markov_head.markov_w1.weight", static_cast<uint64_t>(VOCAB_PADDED) * MTP_MARKOV_RANK);
        mw.dspark.rnn_bias = upload_bf16(arena, up, sf_mtp, "markov_head.rnn_proj.bias", 3 * MTP_MARKOV_RANK);
        mw.dspark.hidden_out_w = nullptr;
        mw.dspark.hidden_out_b = nullptr;

        {
            const uint64_t rnn_bytes = static_cast<uint64_t>(3 * MTP_MARKOV_RANK) * (2 * MTP_MARKOV_RANK + 2 * HIDDEN) * 2;
            upload_raw(up, sf_mtp, "markov_head.rnn_proj.weight", d_stage, rnn_bytes);
            mw.dspark.rnn_w_sp = static_cast<__nv_bfloat16*>(arena.alloc_raw(768ULL * 512 * 2));
            mw.dspark.rnn_w_h = static_cast<__nv_bfloat16*>(arena.alloc_raw(768ULL * HIDDEN * 2));
            mw.dspark.rnn_w_f = static_cast<__nv_bfloat16*>(arena.alloc_raw(768ULL * HIDDEN * 2));
            split_dspark_rnn(d_stage, mw.dspark.rnn_w_sp, mw.dspark.rnn_w_h, mw.dspark.rnn_w_f);
        }

        {
            const TensorInfo* ui = sf_main.lookup("model.llm.unembed.weight");
            const uint8_t* h_unembed = sf_main.data_ptr(ui);
            const size_t row_bytes = static_cast<size_t>(HIDDEN) * 2;

            const size_t hw_bytes = static_cast<size_t>(HIDDEN) * MTP_MARKOV_RANK * 2;
            const size_t hb_bytes = static_cast<size_t>(HIDDEN) * 2;
            void* d_ho_w = nullptr;
            CUDA_CHECK(cudaMalloc(&d_ho_w, hw_bytes + hb_bytes));
            void* d_ho_b = static_cast<uint8_t*>(d_ho_w) + hw_bytes;
            upload_raw(up, sf_mtp, "markov_head.hidden_out.weight", d_ho_w, hw_bytes);
            upload_raw(up, sf_mtp, "markov_head.hidden_out.bias", d_ho_b, hb_bytes);

            uint8_t* h_sub = nullptr;
            const uint8_t* fold_src_host = h_unembed;

            if (has_pruned_draft_vocab) {
                mw.dspark.draft_vocab_size = static_cast<int>(draft_vocab_count);
                mw.dspark.draft_vocab_ids = arena.alloc_many<int32_t>(draft_vocab_count);
                up.upload(mw.dspark.draft_vocab_ids, reinterpret_cast<const uint8_t*>(draft_vocab_host),
                          static_cast<size_t>(draft_vocab_count) * sizeof(int32_t), nullptr);

                mw.dspark.inv_draft_vocab_ids = arena.alloc_many<int32_t>(VOCAB_UNPADDED);
                int32_t* h_inv = alloc_host_zeroed<int32_t>(VOCAB_UNPADDED);
                for (int i = 0; i < VOCAB_UNPADDED; ++i) h_inv[i] = -1;
                for (uint32_t idx = 0; idx < draft_vocab_count; ++idx) {
                    const int32_t tid = draft_vocab_host[idx];
                    SQUID_ASSERT(tid >= 0 && tid < VOCAB_UNPADDED);
                    h_inv[tid] = static_cast<int32_t>(idx);
                }
                up.upload(mw.dspark.inv_draft_vocab_ids, reinterpret_cast<const uint8_t*>(h_inv),
                          static_cast<size_t>(VOCAB_UNPADDED) * sizeof(int32_t), nullptr);
                free_host(h_inv);

                h_sub = alloc_host_zeroed<uint8_t>(static_cast<size_t>(draft_vocab_count) * row_bytes);
                for (uint32_t idx = 0; idx < draft_vocab_count; ++idx) {
                    const size_t src_off = static_cast<size_t>(draft_vocab_host[idx]) * row_bytes;
                    std::memcpy(h_sub + static_cast<size_t>(idx) * row_bytes, h_unembed + src_off, row_bytes);
                }
                mw.dspark.unembed_draft = pack_linear_chunked(
                    arena, up, h_sub, static_cast<size_t>(draft_vocab_count) * row_bytes,
                    draft_vocab_count, HIDDEN, mtp_kind, d_stage, d_amax, nullptr);
                fold_src_host = h_sub;
            } else {
                mw.dspark.draft_vocab_size = VOCAB_UNPADDED;
                mw.dspark.draft_vocab_ids = nullptr;
                mw.dspark.inv_draft_vocab_ids = nullptr;
                mw.dspark.unembed_draft = mw.unembed;
            }

            mw.dspark.fold_b = arena.alloc_many<float>(draft_vocab_count);
            void* d_fold_bf16 = nullptr;
            const size_t fold_bytes = static_cast<size_t>(draft_vocab_count) * MTP_MARKOV_RANK * 2;
            CUDA_CHECK(cudaMalloc(&d_fold_bf16, fold_bytes));
            auto* fold_out_u8 = static_cast<uint8_t*>(d_fold_bf16);

            constexpr uint32_t kFoldChunkRows = 16384;
            if (fuse_unembed_with_dspark_fold) {
                // Fused unembed NVFP4 quantization + DSpark fold (eliminates 922 MiB redundant H2D upload!)
                auto* dst_w = arena.alloc_many<uint8_t>(static_cast<uint64_t>(VOCAB_UNPADDED) * (HIDDEN / 2));
                auto* dst_s = arena.alloc_many<uint8_t>(static_cast<uint64_t>(VOCAB_UNPADDED) * (HIDDEN / 16));
                mw.unembed.kind = WeightKind::NVFP4;
                mw.unembed.rows = VOCAB_UNPADDED;
                mw.unembed.cols = HIDDEN;
                mw.unembed.data = dst_w;
                mw.unembed.scales = dst_s;
                mw.unembed.row_scales = nullptr;

                CUDA_CHECK(cudaMemsetAsync(d_amax, 0, sizeof(float), nullptr));
                for (uint32_t r0 = 0; r0 < draft_vocab_count; ) {
                    const uint32_t rc = std::min(kFoldChunkRows, draft_vocab_count - r0);
                    up.upload(d_stage, fold_src_host + static_cast<size_t>(r0) * row_bytes,
                              static_cast<size_t>(rc) * row_bytes, &sf_main);
                    launch_bf16_amax_accum(d_stage, static_cast<size_t>(rc) * HIDDEN, d_amax, nullptr);
                    r0 += rc;
                }
                float h_amax = 0.0f;
                CUDA_CHECK(cudaMemcpy(&h_amax, d_amax, sizeof(float), cudaMemcpyDeviceToHost));
                const float gscale = std::max(h_amax / (448.0f * 6.0f), 1e-12f);
                mw.unembed.global_scale = gscale;
                mw.dspark.unembed_draft = mw.unembed;

                for (uint32_t r0 = 0; r0 < draft_vocab_count; ) {
                    const uint32_t rc = std::min(kFoldChunkRows, draft_vocab_count - r0);
                    up.upload(d_stage, fold_src_host + static_cast<size_t>(r0) * row_bytes,
                              static_cast<size_t>(rc) * row_bytes, &sf_main);
                    launch_pack_bf16_to_nvfp4_with_scale(
                        d_stage,
                        dst_w + static_cast<size_t>(r0) * (HIDDEN / 2),
                        dst_s + static_cast<size_t>(r0) * (HIDDEN / 16),
                        gscale, static_cast<int32_t>(rc), HIDDEN, nullptr);
                    compute_dspark_fold(
                        d_stage, d_ho_w, d_ho_b,
                        mw.dspark.fold_b + r0,
                        fold_out_u8 + static_cast<size_t>(r0) * MTP_MARKOV_RANK * 2,
                        static_cast<int32_t>(rc), HIDDEN, MTP_MARKOV_RANK);
                    r0 += rc;
                }
            } else {
                for (uint32_t r0 = 0; r0 < draft_vocab_count; ) {
                    const uint32_t rc = std::min(kFoldChunkRows, draft_vocab_count - r0);
                    up.upload(d_stage, fold_src_host + static_cast<size_t>(r0) * row_bytes,
                              static_cast<size_t>(rc) * row_bytes, has_pruned_draft_vocab ? nullptr : &sf_main);
                    compute_dspark_fold(
                        d_stage, d_ho_w, d_ho_b,
                        mw.dspark.fold_b + r0,
                        fold_out_u8 + static_cast<size_t>(r0) * MTP_MARKOV_RANK * 2,
                        static_cast<int32_t>(rc), HIDDEN, MTP_MARKOV_RANK);
                    r0 += rc;
                }
            }
            mw.dspark.fold_w = pack_linear(arena, d_fold_bf16, draft_vocab_count, MTP_MARKOV_RANK, mtp_kind, d_amax);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_fold_bf16));
            CUDA_CHECK(cudaFree(d_ho_w));
            if (h_sub) free_host(h_sub);

            printf("[dspark] Draft vocab size=%u (%s)\n", draft_vocab_count, has_pruned_draft_vocab ? "pruned" : "full");
            std::fflush(stdout);
        }

        auto* stage_u8 = static_cast<uint8_t*>(d_stage);
        for (int ml = 0; ml < MTP_LAYERS; ++ml) {
            DsparkLayerWeights& dl = mw.dspark.layers[ml];
            dl.is_local = is_mtp_local_layer(ml);
            auto upload_mtp = [&](const char* fmt, uint64_t elems) -> __nv_bfloat16* {
                int len = std::snprintf(name_buf, sizeof(name_buf), fmt, ml);
                return upload_bf16(arena, up, sf_mtp, std::string_view(name_buf, static_cast<size_t>(len)), elems);
            };
            auto stage_mtp = [&](const char* fmt, void* dst, uint64_t bytes) {
                int len = std::snprintf(name_buf, sizeof(name_buf), fmt, ml);
                upload_raw(up, sf_mtp, std::string_view(name_buf, static_cast<size_t>(len)), dst, bytes);
            };

            dl.input_norm = upload_mtp("layers.%d.input_layernorm.weight", HIDDEN);
            dl.post_norm = upload_mtp("layers.%d.post_attention_layernorm.weight", HIDDEN);
            dl.dflare_norm = upload_mtp("layers.%d.dflare_norm.weight", HIDDEN);
            dl.q_norm = upload_mtp("layers.%d.self_attn.q_norm.weight", HEAD_DIM);
            dl.k_norm = upload_mtp("layers.%d.self_attn.k_norm.weight", HEAD_DIM);
            dl.k_norm_ctx = upload_mtp("layers.%d.self_attn.k_norm_ctx.weight", HEAD_DIM);

            const uint64_t qb = static_cast<uint64_t>(Q_DIM) * HIDDEN * 2;
            const uint64_t kb = static_cast<uint64_t>(KV_DIM) * HIDDEN * 2;
            stage_mtp("layers.%d.self_attn.q_proj.weight", stage_u8, qb);
            stage_mtp("layers.%d.self_attn.k_proj.weight", stage_u8 + qb, kb);
            stage_mtp("layers.%d.self_attn.v_proj.weight", stage_u8 + qb + kb, kb);
            dl.w_qkv = pack_linear(arena, d_stage, DSPARK_QKV_OUT_DIM, HIDDEN, mtp_kind, d_amax);

            stage_mtp("layers.%d.self_attn.o_proj.weight", stage_u8, static_cast<uint64_t>(HIDDEN) * Q_DIM * 2);
            dl.w_o = pack_linear(arena, d_stage, HIDDEN, Q_DIM, mtp_kind, d_amax);

            const uint64_t gb = static_cast<uint64_t>(MTP_INTERMEDIATE) * HIDDEN * 2;
            stage_mtp("layers.%d.mlp.gate_proj.weight", stage_u8, gb);
            stage_mtp("layers.%d.mlp.up_proj.weight", stage_u8 + gb, gb);
            dl.w_gu = pack_linear(arena, d_stage, 2 * MTP_INTERMEDIATE, HIDDEN, mtp_kind, d_amax);

            stage_mtp("layers.%d.mlp.down_proj.weight", stage_u8, static_cast<uint64_t>(HIDDEN) * MTP_INTERMEDIATE * 2);
            dl.w_down = pack_linear(arena, d_stage, HIDDEN, MTP_INTERMEDIATE, mtp_kind, d_amax);

            stage_mtp("layers.%d.self_attn.k_proj_ctx.weight", stage_u8, kb);
            stage_mtp("layers.%d.self_attn.v_proj_ctx.weight", stage_u8 + kb, kb);
            dl.w_kv_ctx = pack_linear(arena, d_stage, 2 * KV_DIM, HIDDEN, mtp_kind, d_amax);
        }
        mw.dspark.loaded = true;
    }

    // --- Ablation Weights ---
    if (has_ablation) {
        constexpr size_t kDirElems = static_cast<size_t>(NUM_VISITS) * HIDDEN;
        const bool plain_layout = sf_abl.lookup_optional("attn_w") != nullptr;
        float* dir_host = alloc_host_zeroed<float>(kDirElems);
        float w_host[NUM_VISITS]{};
        float c_host[NUM_VISITS]{};
        float tau_host[NUM_VISITS]{};
        float params_host[NUM_VISITS * 4]{};

        float gate_val = 0.0f;
        if (!plain_layout) read_f32_or_bf16(sf_abl, "gate", &gate_val, 1);
        const float floor_val = (!plain_layout && gate_val != 0.0f) ? 0.0f : -FLT_MAX;

        // Attention ablation
        read_f32_or_bf16(sf_abl, "attn_dirs", dir_host, kDirElems);
        normalize_rows_f32(dir_host, NUM_VISITS, HIDDEN);
        ablation.attn_dirs = arena.alloc_many<float>(kDirElems);
        CUDA_CHECK(cudaMemcpy(ablation.attn_dirs, dir_host, kDirElems * 4, cudaMemcpyHostToDevice));

        read_f32_or_bf16(sf_abl, plain_layout ? "attn_w" : "attn_weight", w_host, NUM_VISITS);
        if (plain_layout) std::memset(c_host, 0, sizeof(c_host));
        else read_f32_or_bf16(sf_abl, "attn_offset", c_host, NUM_VISITS);
        if (has_ablation_detectors) {
            read_f32_or_bf16(sf_abl, "attn_det", dir_host, kDirElems);
            ablation.attn_det = arena.alloc_many<float>(kDirElems);
            CUDA_CHECK(cudaMemcpy(ablation.attn_det, dir_host, kDirElems * 4, cudaMemcpyHostToDevice));
            read_f32_or_bf16(sf_abl, "attn_threshold", tau_host, NUM_VISITS);
        } else {
            std::memset(tau_host, 0, sizeof(tau_host));
        }
        for (int v = 0; v < NUM_VISITS; ++v) {
            ablation.attn_nonzero[v] = (w_host[v] != 0.0f);
            params_host[4 * v + 0] = w_host[v];
            params_host[4 * v + 1] = c_host[v];
            params_host[4 * v + 2] = floor_val;
            params_host[4 * v + 3] = tau_host[v];
        }
        ablation.attn_params = arena.alloc_many<float>(NUM_VISITS * 4);
        CUDA_CHECK(cudaMemcpy(ablation.attn_params, params_host, sizeof(params_host), cudaMemcpyHostToDevice));

        // MLP ablation
        read_f32_or_bf16(sf_abl, "mlp_dirs", dir_host, kDirElems);
        normalize_rows_f32(dir_host, NUM_VISITS, HIDDEN);
        ablation.mlp_dirs = arena.alloc_many<float>(kDirElems);
        CUDA_CHECK(cudaMemcpy(ablation.mlp_dirs, dir_host, kDirElems * 4, cudaMemcpyHostToDevice));

        read_f32_or_bf16(sf_abl, plain_layout ? "mlp_w" : "mlp_weight", w_host, NUM_VISITS);
        if (plain_layout) std::memset(c_host, 0, sizeof(c_host));
        else read_f32_or_bf16(sf_abl, "mlp_offset", c_host, NUM_VISITS);
        if (has_ablation_detectors) {
            read_f32_or_bf16(sf_abl, "mlp_det", dir_host, kDirElems);
            ablation.mlp_det = arena.alloc_many<float>(kDirElems);
            CUDA_CHECK(cudaMemcpy(ablation.mlp_det, dir_host, kDirElems * 4, cudaMemcpyHostToDevice));
            read_f32_or_bf16(sf_abl, "mlp_threshold", tau_host, NUM_VISITS);
        } else {
            std::memset(tau_host, 0, sizeof(tau_host));
        }
        for (int v = 0; v < NUM_VISITS; ++v) {
            ablation.mlp_nonzero[v] = (w_host[v] != 0.0f);
            params_host[4 * v + 0] = w_host[v];
            params_host[4 * v + 1] = c_host[v];
            params_host[4 * v + 2] = floor_val;
            params_host[4 * v + 3] = tau_host[v];
        }
        ablation.mlp_params = arena.alloc_many<float>(NUM_VISITS * 4);
        CUDA_CHECK(cudaMemcpy(ablation.mlp_params, params_host, sizeof(params_host), cudaMemcpyHostToDevice));

        free_host(dir_host);
        ablation.active = true;
        ablation.has_detectors = has_ablation_detectors;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    up.deinit();
    CUDA_CHECK(cudaFree(d_amax));
    CUDA_CHECK(cudaFree(d_stage));
    if (draft_vocab_host) free_host(draft_vocab_host);
    if (has_ablation) sf_abl.close_file();
    if (has_mtp) sf_mtp.close_file();
    sf_main.close_file();

    arena.assert_exact_fill();

    const double elapsed_sec = static_cast<double>(clock_monotonic_ns() - start_ns) / 1e9;
    const double vram_gib = static_cast<double>(arena.capacity_bytes) / (1024.0 * 1024.0 * 1024.0);
    printf("[weights] Loaded mode=%s in %.2fs | VRAM=%.2f GiB (1 arena, %u tensors) | MTP=%s | Ablation=%s\n",
           quant_mode_name(quant_mode), elapsed_sec, vram_gib, arena.allocation_count,
           mw.dspark.loaded ? "on" : "off",
           !ablation.active ? "off" : (ablation.has_detectors ? "gated-lda" : "plain"));
    std::fflush(stdout);
    return true;
}

} // namespace squid
