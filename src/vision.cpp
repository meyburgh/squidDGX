#include "vision.hpp"
#include "tiger.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_FAILURE_STRINGS
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-value"
#endif
#include "stb_image.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <curl/curl.h>
#include <webp/decode.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace squid {

void PreprocessedImage::deinit() {
    if (patches) {
        free_host(patches);
        patches = nullptr;
    }
    grid_rows = 0;
    grid_cols = 0;
    num_tokens = 0;
    num_patches = 0;
}

namespace {

constexpr size_t kMaxImageFileBytes = 32 * 1024 * 1024;

inline uint16_t float_to_bf16_bits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    // Round-to-nearest-even (matches CUDA __float2bfloat16)
    const uint32_t rounding_bias = 0x00007FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>((u + rounding_bias) >> 16);
}

bool is_webp_header(const uint8_t* data, size_t len) {
    return len >= 12 &&
           data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F' &&
           data[8] == 'W' && data[9] == 'E' && data[10] == 'B' && data[11] == 'P';
}

bool decode_base64(std::string_view in, uint8_t** out_buf, size_t* out_len) {
    static constexpr int8_t kDec[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,62,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,63,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    };
    const size_t max_out = (in.size() / 4) * 3 + 4;
    if (max_out > kMaxImageFileBytes) return false;
    auto* dst = alloc_host_zeroed<uint8_t>(max_out);
    size_t len = 0;
    uint32_t acc = 0;
    int bits = 0;
    for (char ch : in) {
        const int8_t v = kDec[static_cast<uint8_t>(ch)];
        if (v < 0) continue; // ignore whitespace and '=' padding
        acc = (acc << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            dst[len++] = static_cast<uint8_t>((acc >> bits) & 0xFFu);
        }
    }
    if (len == 0) {
        free_host(dst);
        return false;
    }
    *out_buf = dst;
    *out_len = len;
    return true;
}

struct CurlDownloadBuf {
    uint8_t* data = nullptr;
    size_t len = 0;
    size_t cap = 0;
    bool overflow = false;
};

size_t curl_write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* b = static_cast<CurlDownloadBuf*>(userdata);
    const size_t bytes = size * nmemb;
    if (b->overflow || b->len + bytes > b->cap) {
        b->overflow = true;
        return 0;
    }
    std::memcpy(b->data + b->len, ptr, bytes);
    b->len += bytes;
    return bytes;
}

bool fetch_http_url(std::string_view url, uint8_t** out_buf, size_t* out_len) {
    static const bool curl_inited = (curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK);
    if (!curl_inited || url.size() >= 4096) return false;
    char url_z[4096];
    std::memcpy(url_z, url.data(), url.size());
    url_z[url.size()] = '\0';

    CURL* curl = curl_easy_init();
    if (!curl) return false;

    CurlDownloadBuf dbuf{};
    dbuf.data = alloc_host_zeroed<uint8_t>(kMaxImageFileBytes);
    dbuf.cap = kMaxImageFileBytes;

    curl_easy_setopt(curl, CURLOPT_URL, url_z);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &dbuf);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "squidDGX/1.0");
    curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "http,https");

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK || dbuf.overflow || dbuf.len == 0 || http_code < 200 || http_code >= 300) {
        free_host(dbuf.data);
        return false;
    }
    *out_buf = dbuf.data;
    *out_len = dbuf.len;
    return true;
}

bool read_local_file(std::string_view path, uint8_t** out_buf, size_t* out_len) {
    if (path.size() >= 1024) return false;
    char pz[1024];
    std::memcpy(pz, path.data(), path.size());
    pz[path.size()] = '\0';

    int fd = ::open(pz, O_RDONLY);
    if (fd < 0) return false;
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0 || static_cast<size_t>(st.st_size) > kMaxImageFileBytes) {
        ::close(fd);
        return false;
    }
    const size_t nbytes = static_cast<size_t>(st.st_size);
    auto* buf = alloc_host_zeroed<uint8_t>(nbytes);
    size_t got = 0;
    while (got < nbytes) {
        ssize_t r = ::read(fd, buf + got, nbytes - got);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) {
            free_host(buf);
            ::close(fd);
            return false;
        }
        got += static_cast<size_t>(r);
    }
    ::close(fd);
    *out_buf = buf;
    *out_len = nbytes;
    return true;
}

// Bilinear downscale of RGB uint8 image if native grid exceeds VIS_MAX_TOKENS_PER_IMG.
uint8_t* resize_rgb_bilinear(const uint8_t* src, int sw, int sh, int dw, int dh) {
    auto* dst = alloc_host_zeroed<uint8_t>(static_cast<size_t>(dw) * dh * 3);
    const float scale_x = static_cast<float>(sw) / static_cast<float>(dw);
    const float scale_y = static_cast<float>(sh) / static_cast<float>(dh);
    for (int y = 0; y < dh; ++y) {
        const float sy = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, sh - 1);
        const int y1 = std::clamp(y0 + 1, 0, sh - 1);
        const float fy = std::clamp(sy - std::floor(sy), 0.0f, 1.0f);
        for (int x = 0; x < dw; ++x) {
            const float sx = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, sw - 1);
            const int x1 = std::clamp(x0 + 1, 0, sw - 1);
            const float fx = std::clamp(sx - std::floor(sx), 0.0f, 1.0f);
            const uint8_t* p00 = src + (static_cast<size_t>(y0) * sw + x0) * 3;
            const uint8_t* p01 = src + (static_cast<size_t>(y0) * sw + x1) * 3;
            const uint8_t* p10 = src + (static_cast<size_t>(y1) * sw + x0) * 3;
            const uint8_t* p11 = src + (static_cast<size_t>(y1) * sw + x1) * 3;
            uint8_t* dp = dst + (static_cast<size_t>(y) * dw + x) * 3;
            for (int c = 0; c < 3; ++c) {
                const float top = static_cast<float>(p00[c]) * (1.0f - fx) + static_cast<float>(p01[c]) * fx;
                const float bot = static_cast<float>(p10[c]) * (1.0f - fx) + static_cast<float>(p11[c]) * fx;
                const int val = static_cast<int>(std::lround(top * (1.0f - fy) + bot * fy));
                dp[c] = static_cast<uint8_t>(std::clamp(val, 0, 255));
            }
        }
    }
    return dst;
}

} // namespace

bool preprocess_image_bytes(const uint8_t* data, size_t len, PreprocessedImage& out) {
    out.deinit();
    if (!data || len == 0 || len > kMaxImageFileBytes) return false;

    int width = 0, height = 0, orig_comp = 0;
    uint8_t* rgba = nullptr;
    bool from_webp = false;

    if (is_webp_header(data, len)) {
        rgba = WebPDecodeRGBA(data, len, &width, &height);
        orig_comp = 4;
        from_webp = true;
    } else {
        rgba = stbi_load_from_memory(data, static_cast<int>(len), &width, &height, &orig_comp, 4);
    }

    if (!rgba || width <= 0 || height <= 0 || width > 16384 || height > 16384) {
        if (rgba) {
            if (from_webp) WebPFree(rgba);
            else stbi_image_free(rgba);
        }
        return false;
    }

    // Exact Inkling2ImageProcessor alpha compositing onto 8x8 checkerboard (204 / 153):
    // rgb = (rgba[:3] * alpha + bg * (255 - alpha) + 127) // 255
    auto* rgb = alloc_host_zeroed<uint8_t>(static_cast<size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y) {
        const int yr = y / 8;
        for (int x = 0; x < width; ++x) {
            const size_t si = (static_cast<size_t>(y) * width + x) * 4;
            const size_t di = (static_cast<size_t>(y) * width + x) * 3;
            const int a = rgba[si + 3];
            if (a == 255) {
                rgb[di + 0] = rgba[si + 0];
                rgb[di + 1] = rgba[si + 1];
                rgb[di + 2] = rgba[si + 2];
            } else {
                const int bg = (((yr + x / 8) & 1) == 0) ? 204 : 153;
                const int inv_a = 255 - a;
                rgb[di + 0] = static_cast<uint8_t>((rgba[si + 0] * a + bg * inv_a + 127) / 255);
                rgb[di + 1] = static_cast<uint8_t>((rgba[si + 1] * a + bg * inv_a + 127) / 255);
                rgb[di + 2] = static_cast<uint8_t>((rgba[si + 2] * a + bg * inv_a + 127) / 255);
            }
        }
    }

    if (from_webp) WebPFree(rgba);
    else stbi_image_free(rgba);

    // Cap grid tokens to VIS_MAX_TOKENS_PER_IMG if an ultra-high-res image is submitted.
    int rows = (height + VIS_TILE_SIZE - 1) / VIS_TILE_SIZE;
    int cols = (width + VIS_TILE_SIZE - 1) / VIS_TILE_SIZE;
    if (rows * cols > VIS_MAX_TOKENS_PER_IMG) {
        const double ratio = std::sqrt(static_cast<double>(VIS_MAX_TOKENS_PER_IMG) / static_cast<double>(rows * cols));
        const int nw = std::max(1, static_cast<int>(std::floor(width * ratio)));
        const int nh = std::max(1, static_cast<int>(std::floor(height * ratio)));
        uint8_t* scaled = resize_rgb_bilinear(rgb, width, height, nw, nh);
        free_host(rgb);
        rgb = scaled;
        width = nw;
        height = nh;
        rows = (height + VIS_TILE_SIZE - 1) / VIS_TILE_SIZE;
        cols = (width + VIS_TILE_SIZE - 1) / VIS_TILE_SIZE;
    }

    const int n_tok = rows * cols;
    const int n_patch = n_tok * VIS_MERGE_SIZE * VIS_MERGE_SIZE;
    auto* patches = alloc_host_zeroed<__nv_bfloat16>(static_cast<size_t>(n_patch) * VIS_PATCH_DIM);
    auto* patches_u16 = reinterpret_cast<uint16_t*>(patches);

    // Precompute BF16 LUT for uint8 [0..255] and padded fill_value=-1.0f:
    // norm(p) = (p / 255.0f - 0.5f) / 0.5f = p * (2.0f / 255.0f) - 1.0f
    uint16_t lut_bf16[256];
    for (int v = 0; v < 256; ++v) {
        lut_bf16[v] = float_to_bf16_bits(static_cast<float>(v) * (2.0f / 255.0f) - 1.0f);
    }
    const uint16_t pad_bf16 = float_to_bf16_bits((-1.0f / 255.0f - 0.5f) / 0.5f);

    // Pack directly in spatial-merge-block order:
    // for each token (br, bc), sub-patch (in_r, in_c) in 2x2, pixel (py, px) in 20x20, channel (0..2).
    for (int br = 0; br < rows; ++br) {
        for (int bc = 0; bc < cols; ++bc) {
            const int tok = br * cols + bc;
            for (int in_r = 0; in_r < VIS_MERGE_SIZE; ++in_r) {
                for (int in_c = 0; in_c < VIS_MERGE_SIZE; ++in_c) {
                    const int p_idx = tok * 4 + in_r * 2 + in_c;
                    uint16_t* dst = patches_u16 + static_cast<size_t>(p_idx) * VIS_PATCH_DIM;
                    const int y_base = br * VIS_TILE_SIZE + in_r * VIS_PATCH_SIZE;
                    const int x_base = bc * VIS_TILE_SIZE + in_c * VIS_PATCH_SIZE;
                    size_t off = 0;
                    for (int py = 0; py < VIS_PATCH_SIZE; ++py) {
                        const int y = y_base + py;
                        const bool y_in = (y < height);
                        const uint8_t* row_ptr = y_in ? (rgb + static_cast<size_t>(y) * width * 3) : nullptr;
                        for (int px = 0; px < VIS_PATCH_SIZE; ++px) {
                            const int x = x_base + px;
                            if (y_in && x < width) {
                                const uint8_t* pix = row_ptr + static_cast<size_t>(x) * 3;
                                dst[off + 0] = lut_bf16[pix[0]];
                                dst[off + 1] = lut_bf16[pix[1]];
                                dst[off + 2] = lut_bf16[pix[2]];
                            } else {
                                dst[off + 0] = pad_bf16;
                                dst[off + 1] = pad_bf16;
                                dst[off + 2] = pad_bf16;
                            }
                            off += 3;
                        }
                    }
                }
            }
        }
    }

    free_host(rgb);
    out.grid_rows = rows;
    out.grid_cols = cols;
    out.num_tokens = n_tok;
    out.num_patches = n_patch;
    out.patches = patches;
    return true;
}

bool preprocess_image_source(std::string_view src, PreprocessedImage& out) {
    out.deinit();
    if (src.empty()) return false;

    uint8_t* raw = nullptr;
    size_t raw_len = 0;
    bool ok = false;

    if (src.size() > 5 && src.substr(0, 5) == "data:") {
        const size_t comma = src.find(',');
        if (comma == std::string_view::npos) return false;
        const std::string_view meta = src.substr(5, comma - 5);
        const std::string_view payload = src.substr(comma + 1);
        if (meta.find(";base64") != std::string_view::npos) {
            ok = decode_base64(payload, &raw, &raw_len);
        }
    } else if ((src.size() > 7 && src.substr(0, 7) == "http://") ||
               (src.size() > 8 && src.substr(0, 8) == "https://")) {
        ok = fetch_http_url(src, &raw, &raw_len);
    } else if (src.size() > 7 && src.substr(0, 7) == "file://") {
        ok = read_local_file(src.substr(7), &raw, &raw_len);
    } else {
        ok = read_local_file(src, &raw, &raw_len);
    }

    if (!ok || !raw) return false;
    const bool decoded = preprocess_image_bytes(raw, raw_len, out);
    free_host(raw);
    return decoded;
}

} // namespace squid
