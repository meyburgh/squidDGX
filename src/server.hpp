#pragma once

// TigerStyle Zero-Malloc-Per-Request OpenAI HTTP/1.1 + SSE Server.

#include "engine.hpp"
#include "tokenizer.hpp"
#include <cstdint>
#include <string_view>

namespace squid {

struct ServerConfig {
    std::string_view host = "0.0.0.0";
    uint16_t port = 8000;
    uint16_t semif_port = 8897;
    uint32_t semif_max_tokens = 16384;
    uint32_t prefix_sessions = 3;   // conversations parked in host RAM besides the live one
    std::string_view model_name = "inkling-2-mini";
    // Weights-specific identity reported as model.weights_id on the SemIf endpoint (see compute_weights_id).
    std::string_view weights_id = "Inkling-2-Mini@unknown";
    std::string_view api_key = "";
    std::string_view default_effort = "none";
    uint32_t default_mtp = MTP_MAX_DRAFTS;
    float default_temperature = 1.0f;
    float default_top_p = 0.95f;
    int32_t default_top_k = 0;
    uint32_t context_tokens_max = 262144;
    uint32_t max_output_tokens = 65536;
    bool raw_thinking = false;
    bool keep_history_thinking = false;
};

// Builds a weights identity that changes iff the served weights change:
//   "Inkling-2-Mini.sha256x64m-<16hex>.<mode>[.abl-sha256x64m-<16hex>]"  (chars limited to [A-Za-z0-9.-])
// sha256x64m = SHA-256 over the concatenated hex SHA-256 digests of consecutive 64 MiB chunks
// of the file (parallel, so startup stays at disk speed). Mode is the weight quantization
// (nvfp4|fp8|exact), which changes the effective weights. Returns false on I/O failure.
bool compute_weights_id(std::string_view model_directory, std::string_view ablation_path,
                        std::string_view mode, char* out, size_t out_cap);

int run_openai_server(Engine& engine, const Tokenizer& tokenizer, const ServerConfig& config);

// Exposed for unit testing UTF-8 boundary handling
size_t valid_utf8_prefix_len(std::string_view s);

} // namespace squid
