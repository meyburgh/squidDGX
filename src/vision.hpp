#pragma once

// Inkling-2-Mini Image Decoder & Preprocessor (Inkling2ImageProcessor):
// - Base64 data URIs, local file paths, and HTTP/HTTPS URLs (via libcurl)
// - JPEG/PNG/BMP/GIF/TGA/PNM/PSD/HDR (stb_image) + WebP (libwebp)
// - Exact 8x8 checkerboard alpha compositing (204/153), -1.0 grid padding,
//   [-1, 1] normalization, and BF16 spatial-merge-block patch packing [T*4, 1200].

#include "config.hpp"
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace squid {

struct PreprocessedImage {
    int grid_rows = 0;             // ceil(height / 40)
    int grid_cols = 0;             // ceil(width / 40)
    int num_tokens = 0;            // grid_rows * grid_cols
    int num_patches = 0;           // num_tokens * 4
    __nv_bfloat16* patches = nullptr; // host-allocated [num_patches * VIS_PATCH_DIM]

    void deinit();
};

// Decode & preprocess raw image file bytes (PNG, JPEG, WebP, GIF, BMP, ...).
bool preprocess_image_bytes(const uint8_t* data, size_t len, PreprocessedImage& out);

// Load & preprocess an image from a data: URI, file:// path, plain file path, or http(s):// URL.
bool preprocess_image_source(std::string_view src, PreprocessedImage& out);

} // namespace squid
