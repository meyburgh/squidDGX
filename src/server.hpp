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
    std::string_view model_name = "inkling-2-mini";
    std::string_view api_key = "";
    std::string_view default_effort = "none";
    uint32_t default_mtp = 3;
    float default_temperature = 1.0f;
    float default_top_p = 0.95f;
    int32_t default_top_k = 0;
    uint32_t context_tokens_max = 65536;
    bool raw_thinking = false;
    bool keep_history_thinking = false;
};

int run_openai_server(Engine& engine, const Tokenizer& tokenizer, const ServerConfig& config);

// Exposed for unit testing UTF-8 boundary handling
size_t valid_utf8_prefix_len(std::string_view s);

} // namespace squid
