#pragma once

// TigerStyle two-pass ModelWeights loader directly populating squid::ModelWeights.
// Pass 1 computes exact 256-byte-aligned VRAM capacity; Pass 2 allocates ONE VRAMArena.

#include "config.hpp"
#include "vram_arena.hpp"
#include "../cuda/weights.hpp"
#include <cstdint>
#include <string_view>

namespace squid {

struct AblationWeights {
    bool active = false;
    bool has_detectors = false;
    float* attn_dirs = nullptr;   // [48 * 2304]
    float* mlp_dirs = nullptr;    // [48 * 2304]
    float* attn_det = nullptr;    // [48 * 2304] optional
    float* mlp_det = nullptr;     // [48 * 2304] optional
    float* attn_params = nullptr; // [48 * 4]
    float* mlp_params = nullptr;  // [48 * 4]
    bool attn_nonzero[NUM_VISITS]{};
    bool mlp_nonzero[NUM_VISITS]{};
};

struct LoadedModel {
    ModelWeights mw{};
    VRAMArena arena{};
    AblationWeights ablation{};

    bool load(
        std::string_view model_directory,
        QuantMode quant_mode,
        bool load_mtp,
        std::string_view ablation_path);

    void deinit();
};

} // namespace squid
