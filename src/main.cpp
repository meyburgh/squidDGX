// squidDGX — Pure C++20 + Blackwell sm_121a CUDA Inference Engine & OpenAI Server.
// Zero dynamic allocations post-init(), single-arena VRAM weights & runtime, always-on assertions.

#include "config.hpp"
#include "engine.hpp"
#include "server.hpp"
#include "tiger.hpp"
#include "tokenizer.hpp"
#include "vision.hpp"
#include "weights.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace {

constexpr size_t kBenchRunsMax = 64;
constexpr size_t kCliTokensMax = 1048576;

void print_usage() {
    static constexpr std::string_view kUsage =
        "squidDGX — C++20 + Blackwell CUDA Inference Engine & OpenAI Server for Inkling-2-Mini\n\n"
        "Usage:\n"
        "  squidDGX serve    <model_dir> [--host 0.0.0.0] [--port 8000] [--mode nvfp4|fp8|exact] [--kv bf16|auto|fp8|q4] [--ctx 262144] [--max-tokens 65536] [--mtp 7] [--spec-threshold 0.75] [--temp 1.0] [--top-p 0.95] [--top-k 0] [--keep-history-thinking] [--ablation PATH] [--model-name inkling-2-mini] [--effort none] [--api-key KEY]\n"
        "  squidDGX generate <model_dir> (--prompt TEXT | --prompt-file PATH | --tokens CSV | --tokens-file PATH) [--image PATH]... [--mode nvfp4] [--kv bf16] [--mtp 7] [--spec-threshold 0.75] [--temp 0.0] [--top-p 0.95] [--top-k 0] [--seed N] [--ablation PATH] [--max-tokens 65536] [--ctx 262144]\n"
        "  squidDGX bench    <model_dir> (--prompt TEXT | --prompt-file PATH | --tokens CSV) [--mode nvfp4] [--kv bf16] [--mtp 7] [--spec-threshold 0.75] [--temp 0.0] [--top-p 0.95] [--ablation PATH] [--max-tokens 128] [--runs 6] [--ctx 262144]\n"
        "  squidDGX longctx  <model_dir> [--mode nvfp4|fp8|exact] [--kv bf16|fp8|q4] [--ctx 262144]\n"
        "  squidDGX logits   <model_dir> (--prompt TEXT | --prompt-file PATH | --tokens-file PATH) --out FILE [--mode nvfp4] [--kv bf16] [--ctx 262144]\n"
        "  squidDGX kld      <model_dir> (--prompt TEXT | --prompt-file PATH | --tokens-file PATH) --ref FILE [--mode nvfp4|fp8|exact] [--kv bf16|auto|fp8|q4] [--ctx 262144]\n"
        "  squidDGX tokenize <model_dir> (--prompt TEXT | --prompt-file PATH) [--raw]\n";
    squid::write_fd_all(2, kUsage);
}

squid::QuantMode parse_mode(std::string_view s) {
    if (s == "exact") return squid::QuantMode::Exact;
    if (s == "fp8") return squid::QuantMode::FP8;
    if (s == "nvfp4" || s == "nv4fp") return squid::QuantMode::NVFP4;
    fprintf(stderr, "Unknown mode '%.*s' (expected exact|fp8|nvfp4)\n", static_cast<int>(s.size()), s.data());
    std::exit(1);
}

size_t parse_csv_tokens(std::string_view csv, int32_t* out, size_t cap) {
    size_t count = 0;
    size_t pos = 0;
    while (pos < csv.size() && count < cap) {
        while (pos < csv.size() && (csv[pos] == ',' || csv[pos] == ' ' || csv[pos] == '\n' || csv[pos] == '\r' || csv[pos] == '\t')) ++pos;
        if (pos >= csv.size()) break;
        const size_t start = pos;
        while (pos < csv.size() && csv[pos] != ',' && csv[pos] != ' ' && csv[pos] != '\n' && csv[pos] != '\r' && csv[pos] != '\t') ++pos;
        const size_t len = pos - start;
        if (len > 0 && len < 32) {
            char tmp[32];
            std::memcpy(tmp, csv.data() + start, len);
            tmp[len] = '\0';
            char* endp = nullptr;
            long v = std::strtol(tmp, &endp, 10);
            if (endp == tmp + len) out[count++] = static_cast<int32_t>(v);
        }
    }
    return count;
}

bool read_file_into(std::string_view path, char* buf, size_t cap, std::string_view* out) {
    char pz[1024];
    if (path.size() >= sizeof(pz)) return false;
    std::memcpy(pz, path.data(), path.size());
    pz[path.size()] = '\0';
    int fd = ::open(pz, O_RDONLY);
    if (fd < 0) return false;
    size_t total = 0;
    for (uint32_t iter = 0; total < cap && iter < 65536; ++iter) {
        ssize_t n = ::read(fd, buf + total, cap - total);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        total += static_cast<size_t>(n);
    }
    ::close(fd);
    *out = std::string_view(buf, total);
    return true;
}

void print_token_csv(std::string_view prefix, const int32_t* tokens, size_t count) {
    squid::write_fd_all(1, prefix);
    char chunk[4096];
    squid::ByteWriter w(chunk, sizeof(chunk));
    for (size_t idx = 0; idx < count; ++idx) {
        if (w.remaining() < 24) {
            squid::write_fd_all(1, w.slice());
            w.reset();
        }
        w.append_fmt("%s%d", (idx == 0) ? " " : ",", tokens[idx]);
    }
    w.append_byte('\n');
    squid::write_fd_all(1, w.slice());
}

void print_decoded(const squid::Tokenizer& tok, const int32_t* tokens, size_t count) {
    squid::write_fd_all(1, "\n--- DECODED OUTPUT ---\n");
    for (size_t i = 0; i < count; ++i) {
        std::string_view bytes = tok.decode_token_bytes(tokens[i], false);
        if (!bytes.empty()) squid::write_fd_all(1, bytes);
    }
    squid::write_fd_all(1, "\n----------------------\n");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    const std::string_view command(argv[1]);
    const std::string_view model_directory(argv[2]);

    std::string_view tokens_csv = "1,2,3";
    std::string_view tokens_file;
    std::string_view prompt_text;
    std::string_view prompt_file;
    std::string_view mode_str = "nvfp4";
    std::string_view kv_str = "bf16";
    std::string_view host = "0.0.0.0";
    std::string_view model_name = "inkling-2-mini";
    bool model_name_explicit = false;
    std::string_view ablation_path;
    std::string_view out_path;
    std::string_view ref_path;
    std::string_view effort = "none";
    std::string_view api_key;
    if (const char* env_key = std::getenv("SQUID_API_KEY")) api_key = env_key;

    uint16_t port = 8000;
    const bool generation_command = (command == "serve" || command == "generate" || command == "bench");
    uint32_t mtp = generation_command ? static_cast<uint32_t>(squid::MTP_MAX_DRAFTS) : 0;
    float spec_threshold = 0.75f;
    float temperature = (command == "serve") ? 1.0f : 0.0f;
    float top_p = 0.95f;
    int32_t top_k = 0;
    uint64_t seed = 0;
    uint32_t max_tokens = (command == "bench") ? 128u : 65536u;
    uint32_t runs = 6;
    uint32_t ctx = 262144u;
    bool use_graphs = true;
    bool use_prompt_lookup = true;
    bool raw_prompt = false;
    bool raw_thinking = false;
    bool keep_history_thinking = false;
    bool has_tokens_arg = false;
    std::string_view image_paths[16];
    size_t image_count = 0;

    for (int arg_i = 3; arg_i < argc; ++arg_i) {
        const std::string_view arg(argv[arg_i]);
        auto has_val = [&]() { return arg_i + 1 < argc; };
        if (arg == "--tokens" && has_val()) { tokens_csv = argv[++arg_i]; has_tokens_arg = true; }
        else if (arg == "--tokens-file" && has_val()) { tokens_file = argv[++arg_i]; has_tokens_arg = true; }
        else if (arg == "--prompt" && has_val()) { prompt_text = argv[++arg_i]; }
        else if (arg == "--prompt-file" && has_val()) { prompt_file = argv[++arg_i]; }
        else if (arg == "--image" && has_val()) {
            if (image_count < 16) image_paths[image_count++] = argv[++arg_i];
            else ++arg_i;
        }
        else if (arg == "--mode" && has_val()) { mode_str = argv[++arg_i]; }
        else if (arg == "--kv" && has_val()) { kv_str = argv[++arg_i]; }
        else if (arg == "--host" && has_val()) { host = argv[++arg_i]; }
        else if (arg == "--port" && has_val()) { port = static_cast<uint16_t>(std::strtoul(argv[++arg_i], nullptr, 10)); }
        else if (arg == "--model-name" && has_val()) { model_name = argv[++arg_i]; model_name_explicit = true; }
        else if (arg == "--ablation" && has_val()) { ablation_path = argv[++arg_i]; }
        else if (arg == "--out" && has_val()) { out_path = argv[++arg_i]; }
        else if (arg == "--ref" && has_val()) { ref_path = argv[++arg_i]; }
        else if (arg == "--effort" && has_val()) { effort = argv[++arg_i]; }
        else if (arg == "--api-key" && has_val()) { api_key = argv[++arg_i]; }
        else if (arg == "--mtp" && has_val()) { mtp = static_cast<uint32_t>(std::strtoul(argv[++arg_i], nullptr, 10)); }
        else if (arg == "--spec-threshold" && has_val()) {
            float v = std::strtof(argv[++arg_i], nullptr);
            spec_threshold = std::isfinite(v) ? v : 0.75f;
        } else if (arg == "--temp" && has_val()) {
            float v = std::strtof(argv[++arg_i], nullptr);
            temperature = (std::isfinite(v) && v >= 0.0f) ? v : 0.0f;
        } else if (arg == "--top-p" && has_val()) {
            float v = std::strtof(argv[++arg_i], nullptr);
            top_p = (std::isfinite(v) && v > 0.0f && v <= 1.0f) ? v : 0.95f;
        } else if (arg == "--top-k" && has_val()) {
            top_k = static_cast<int32_t>(std::strtol(argv[++arg_i], nullptr, 10));
        } else if (arg == "--seed" && has_val()) {
            seed = std::strtoull(argv[++arg_i], nullptr, 10);
        } else if (arg == "--max-tokens" && has_val()) {
            max_tokens = static_cast<uint32_t>(std::strtoul(argv[++arg_i], nullptr, 10));
        } else if (arg == "--runs" && has_val()) {
            runs = static_cast<uint32_t>(std::strtoul(argv[++arg_i], nullptr, 10));
        } else if (arg == "--ctx" && has_val()) {
            ctx = static_cast<uint32_t>(std::strtoul(argv[++arg_i], nullptr, 10));
        } else if (arg == "--no-graph") { use_graphs = false; }
        else if (arg == "--no-prompt-lookup") { use_prompt_lookup = false; }
        else if (arg == "--raw") { raw_prompt = true; }
        else if (arg == "--raw-thinking") { raw_thinking = true; }
        else if (arg == "--keep-history-thinking") { keep_history_thinking = true; }
    }

    char* prompt_file_buf = nullptr;
    if (!prompt_file.empty()) {
        prompt_file_buf = squid::alloc_host_zeroed<char>(32 * 1024 * 1024);
        if (!read_file_into(prompt_file, prompt_file_buf, 32 * 1024 * 1024, &prompt_text)) {
            fprintf(stderr, "Failed to read --prompt-file '%.*s'\n", static_cast<int>(prompt_file.size()), prompt_file.data());
            squid::free_host(prompt_file_buf);
            return 1;
        }
    }

    if (command == "tokenize") {
        squid::Tokenizer tok{};
        if (!tok.load(model_directory)) { squid::free_host(prompt_file_buf); return 1; }
        squid::TokenizerScratch scratch{};
        scratch.init();

        const size_t render_cap = std::max<size_t>(prompt_text.size() + 4096, 65536);
        char* render_buf = squid::alloc_host_zeroed<char>(render_cap);
        const size_t ids_cap = std::min(kCliTokensMax, std::max<size_t>(render_cap, 8192));
        int32_t* ids_buf = squid::alloc_host_zeroed<int32_t>(ids_cap);

        std::string_view input_slice = prompt_text;
        if (!raw_prompt) {
            squid::ByteWriter rw(render_buf, render_cap);
            squid::Tokenizer::render_single_user_prompt(rw, prompt_text, effort);
            input_slice = rw.slice();
        }
        const uint64_t t0 = squid::clock_monotonic_ns();
        const size_t count = tok.encode_into(input_slice, true, scratch, ids_buf, ids_cap);
        const double enc_us = static_cast<double>(squid::clock_monotonic_ns() - t0) / 1000.0;
        printf("[tokenize] count=%zu encode_us=%.1f\n", count, enc_us);
        std::fflush(stdout);
        print_token_csv("IDS:", ids_buf, count);

        squid::free_host(ids_buf);
        squid::free_host(render_buf);
        scratch.deinit();
        tok.deinit();
        squid::free_host(prompt_file_buf);
        return 0;
    }

    const bool need_tokenizer = (command == "serve" || !prompt_text.empty() || image_count > 0);
    squid::Tokenizer tokenizer{};
    if (need_tokenizer && !tokenizer.load(model_directory)) {
        squid::free_host(prompt_file_buf);
        return 1;
    }

    if (ctx < 64 || ctx > kCliTokensMax) {
        fprintf(stderr, "--ctx must be in 64..1048576\n");
        if (need_tokenizer) tokenizer.deinit();
        squid::free_host(prompt_file_buf);
        return 1;
    }

    const squid::QuantMode quant_mode = parse_mode(mode_str);
    const squid::KVFormat kv_format = (kv_str == "bf16") ? squid::KVFormat::BF16
                                    : (kv_str == "fp8")  ? squid::KVFormat::FP8
                                    : (kv_str == "q4")   ? squid::KVFormat::Q4
                                    : (quant_mode == squid::QuantMode::Exact) ? squid::KVFormat::BF16
                                    : (ctx > 458752)     ? squid::KVFormat::Q4
                                                         : squid::KVFormat::FP8;

    const size_t prompt_ids_cap = std::min(kCliTokensMax, std::max<size_t>(static_cast<size_t>(ctx) + 1, 1024));
    int32_t* prompt_ids_buf = squid::alloc_host_zeroed<int32_t>(prompt_ids_cap);
    size_t prompt_len = 0;

    squid::PreprocessedImage cli_images[16]{};
    size_t cli_image_count = 0;
    for (size_t ii = 0; ii < image_count; ++ii) {
        if (!squid::preprocess_image_source(image_paths[ii], cli_images[cli_image_count])) {
            fprintf(stderr, "Failed to load/decode --image '%.*s'\n",
                    static_cast<int>(image_paths[ii].size()), image_paths[ii].data());
            for (size_t jj = 0; jj < cli_image_count; ++jj) cli_images[jj].deinit();
            squid::free_host(prompt_ids_buf);
            if (need_tokenizer) tokenizer.deinit();
            squid::free_host(prompt_file_buf);
            return 1;
        }
        ++cli_image_count;
    }

    if ((!prompt_text.empty() || cli_image_count > 0) && !has_tokens_arg) {
        squid::TokenizerScratch scratch{};
        scratch.init();
        const size_t render_cap = std::max<size_t>(prompt_text.size() + 4096, 65536);
        char* render_buf = squid::alloc_host_zeroed<char>(render_cap);
        if (raw_prompt) {
            prompt_len = tokenizer.encode_into(prompt_text, true, scratch, prompt_ids_buf, prompt_ids_cap);
        } else {
            squid::ByteWriter rw(render_buf, render_cap);
            squid::Tokenizer::emit_thinking_effort(rw, effort);
            prompt_len += tokenizer.encode_into(rw.slice(), true, scratch, prompt_ids_buf + prompt_len, prompt_ids_cap - prompt_len);
            for (size_t ii = 0; ii < cli_image_count; ++ii) {
                prompt_len += tokenizer.encode_into("<|message_user|><|content_image|>", true, scratch,
                                                    prompt_ids_buf + prompt_len, prompt_ids_cap - prompt_len);
                const int n_tok = cli_images[ii].num_tokens;
                for (int it = 0; it < n_tok && prompt_len < prompt_ids_cap; ++it) {
                    prompt_ids_buf[prompt_len++] = squid::IMAGE_TOKEN_ID;
                }
                prompt_len += tokenizer.encode_into("<|end_message|>", true, scratch,
                                                    prompt_ids_buf + prompt_len, prompt_ids_cap - prompt_len);
            }
            if (!prompt_text.empty()) {
                rw.reset();
                rw.append_slice("<|message_user|><|content_text|>");
                rw.append_slice(prompt_text);
                rw.append_slice("<|end_message|>");
                prompt_len += tokenizer.encode_into(rw.slice(), true, scratch, prompt_ids_buf + prompt_len, prompt_ids_cap - prompt_len);
            }
        }
        squid::free_host(render_buf);
        scratch.deinit();
    } else if (!tokens_file.empty()) {
        char* file_buf = squid::alloc_host_zeroed<char>(32 * 1024 * 1024);
        std::string_view loaded;
        if (!read_file_into(tokens_file, file_buf, 32 * 1024 * 1024, &loaded)) {
            fprintf(stderr, "Failed to read --tokens-file '%.*s'\n", static_cast<int>(tokens_file.size()), tokens_file.data());
            squid::free_host(file_buf);
            squid::free_host(prompt_ids_buf);
            if (need_tokenizer) tokenizer.deinit();
            squid::free_host(prompt_file_buf);
            return 1;
        }
        prompt_len = parse_csv_tokens(loaded, prompt_ids_buf, prompt_ids_cap);
        squid::free_host(file_buf);
    } else {
        prompt_len = parse_csv_tokens(tokens_csv, prompt_ids_buf, prompt_ids_cap);
    }

    for (size_t i = 0; i < prompt_len; ++i) {
        if (prompt_ids_buf[i] < 0 || prompt_ids_buf[i] >= squid::VOCAB_PADDED) {
            fprintf(stderr, "Invalid token id %d (valid range: 0..%d)\n", prompt_ids_buf[i], squid::VOCAB_PADDED - 1);
            squid::free_host(prompt_ids_buf);
            if (need_tokenizer) tokenizer.deinit();
            squid::free_host(prompt_file_buf);
            return 1;
        }
    }
    const bool is_prompt_cmd = (command == "logits" || command == "kld" || command == "generate" || command == "bench");
    if (is_prompt_cmd && (prompt_len == 0 || prompt_len >= ctx)) {
        fprintf(stderr, "Prompt length %zu must be in 1..%u (increase --ctx)\n", prompt_len, ctx - 1);
        squid::free_host(prompt_ids_buf);
        if (need_tokenizer) tokenizer.deinit();
        squid::free_host(prompt_file_buf);
        return 1;
    }

    if (!ablation_path.empty() && !model_name_explicit) model_name = "inkling-2-mini-abliterated";

    squid::LoadedModel model{};
    const bool load_mtp = (mtp > 0 || command == "longctx" || command == "serve");
    if (!model.load(model_directory, quant_mode, load_mtp, ablation_path)) {
        squid::free_host(prompt_ids_buf);
        if (need_tokenizer) tokenizer.deinit();
        squid::free_host(prompt_file_buf);
        return 1;
    }

    squid::Engine engine{};
    if (!engine.init(&model, ctx, use_graphs, kv_format)) {
        model.deinit();
        squid::free_host(prompt_ids_buf);
        if (need_tokenizer) tokenizer.deinit();
        squid::free_host(prompt_file_buf);
        return 1;
    }
    engine.spec_threshold = spec_threshold;
    engine.use_prompt_lookup = use_prompt_lookup;
    engine.sample_params = {temperature, top_p, top_k, seed != 0 ? seed : squid::clock_monotonic_ns()};

    int exit_code = 0;

    if (command == "serve") {
        squid::ServerConfig scfg{};
        scfg.host = host;
        scfg.port = port;
        scfg.model_name = model_name;
        scfg.api_key = api_key;
        scfg.default_effort = effort;
        scfg.default_mtp = mtp;
        scfg.default_temperature = temperature;
        scfg.default_top_p = top_p;
        scfg.default_top_k = top_k;
        scfg.context_tokens_max = ctx;
        scfg.max_output_tokens = std::min(std::max(max_tokens, 1u), ctx);
        scfg.raw_thinking = raw_thinking;
        scfg.keep_history_thinking = keep_history_thinking;
        exit_code = squid::run_openai_server(engine, tokenizer, scfg);
    } else if (command == "logits") {
        if (out_path.empty() || prompt_len < 2 || out_path.size() >= 1024) {
            fprintf(stderr, "Usage: squidDGX logits <model_dir> (--prompt TEXT | --prompt-file PATH | --tokens-file PATH) --out <file.bin> [--mode nvfp4] [--kv auto]\n");
            exit_code = 1;
        } else {
            constexpr size_t kVocab = squid::VOCAB_UNPADDED;
            float* chunk_buf = squid::alloc_host_zeroed<float>(static_cast<size_t>(squid::MTP_BLOCK_SIZE) * kVocab);
            char pz[1024];
            std::memcpy(pz, out_path.data(), out_path.size());
            pz[out_path.size()] = '\0';
            int fd = ::open(pz, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd < 0) {
                fprintf(stderr, "Failed to open --out '%s' for writing\n", pz);
                exit_code = 1;
            } else {
                struct WriteCtx {
                    int fd;
                    size_t bytes_written = 0;
                    bool write_err = false;
                    static bool on_chunk(void* p, size_t, const float* logits, size_t cnt) {
                        auto* self = static_cast<WriteCtx*>(p);
                        const size_t nbytes = cnt * squid::VOCAB_UNPADDED * sizeof(float);
                        if (!squid::write_fd_all_checked(self->fd, std::string_view(reinterpret_cast<const char*>(logits), nbytes))) {
                            self->write_err = true;
                            return false;
                        }
                        self->bytes_written += nbytes;
                        return true;
                    }
                } wctx{fd};
                if (!engine.eval_logits_streaming(prompt_ids_buf, prompt_len, chunk_buf, static_cast<size_t>(squid::MTP_BLOCK_SIZE) * kVocab, WriteCtx::on_chunk, &wctx) || wctx.write_err) {
                    fprintf(stderr, "I/O error writing logits to '%s'\n", pz);
                    exit_code = 1;
                } else {
                    printf("[logits] wrote %zu positions x %zu vocab (%.1f MiB) to %s (mode=%s, kv=%s)\n",
                           prompt_len, kVocab, static_cast<double>(wctx.bytes_written) / (1024.0 * 1024.0),
                           pz, squid::quant_mode_name(quant_mode), squid::kv_format_name(kv_format));
                    std::fflush(stdout);
                }
                ::close(fd);
            }
            squid::free_host(chunk_buf);
        }
    } else if (command == "kld") {
        if (ref_path.empty() || prompt_len < 2 || ref_path.size() >= 1024) {
            fprintf(stderr, "Usage: squidDGX kld <model_dir> (--prompt TEXT | --prompt-file PATH | --tokens-file PATH) --ref <ref_logits.bin> [--mode nvfp4] [--kv q4]\n");
            exit_code = 1;
        } else {
            constexpr size_t kVocab = squid::VOCAB_UNPADDED;
            char pz[1024];
            std::memcpy(pz, ref_path.data(), ref_path.size());
            pz[ref_path.size()] = '\0';
            int fd = ::open(pz, O_RDONLY);
            if (fd < 0) {
                fprintf(stderr, "Failed to open --ref '%s'\n", pz);
                exit_code = 1;
            } else {
                float* chunk_buf = squid::alloc_host_zeroed<float>(static_cast<size_t>(squid::MTP_BLOCK_SIZE) * kVocab);
                float* ref_row = squid::alloc_host_zeroed<float>(kVocab);
                double* kld_vals = squid::alloc_host_zeroed<double>(prompt_len);

                struct KldCtx {
                    int fd;
                    size_t n_pos;
                    const int32_t* prompt_ids;
                    float* ref_row;
                    double* kld_vals;
                    double sum_kld = 0.0;
                    double sum_nll_ref = 0.0;
                    double sum_nll_cand = 0.0;
                    size_t nll_cnt = 0;
                    size_t top1_match = 0;
                    bool eof_err = false;

                    static bool on_chunk(void* p, size_t pos_offset, const float* chunk_logits, size_t cnt) {
                        auto* self = static_cast<KldCtx*>(p);
                        constexpr size_t kV = squid::VOCAB_UNPADDED;
                        constexpr size_t kRowBytes = kV * sizeof(float);
                        auto* ref_bytes = reinterpret_cast<uint8_t*>(self->ref_row);
                        for (size_t b = 0; b < cnt; ++b) {
                            const size_t p_i = pos_offset + b;
                            size_t got = 0;
                            while (got < kRowBytes) {
                                ssize_t n = ::read(self->fd, ref_bytes + got, kRowBytes - got);
                                if (n < 0 && errno == EINTR) continue;
                                if (n <= 0) {
                                    fprintf(stderr, "Unexpected EOF reading --ref at pos %zu\n", p_i);
                                    self->eof_err = true;
                                    return false;
                                }
                                got += static_cast<size_t>(n);
                            }
                            const float* c_row = chunk_logits + b * kV;
                            double max_r = -1e30, max_c = -1e30;
                            size_t arg_r = 0, arg_c = 0;
                            for (size_t v = 0; v < kV; ++v) {
                                if (self->ref_row[v] > max_r) { max_r = self->ref_row[v]; arg_r = v; }
                                if (c_row[v] > max_c) { max_c = c_row[v]; arg_c = v; }
                            }
                            if (arg_r == arg_c) self->top1_match += 1;
                            double z_r = 0.0, z_c = 0.0;
                            for (size_t v = 0; v < kV; ++v) {
                                z_r += std::exp(static_cast<double>(self->ref_row[v]) - max_r);
                                z_c += std::exp(static_cast<double>(c_row[v]) - max_c);
                            }
                            const double log_zr = max_r + std::log(z_r);
                            const double log_zc = max_c + std::log(z_c);
                            double kl = 0.0;
                            for (size_t v = 0; v < kV; ++v) {
                                const double log_p = static_cast<double>(self->ref_row[v]) - log_zr;
                                const double log_q = static_cast<double>(c_row[v]) - log_zc;
                                kl += std::exp(log_p) * (log_p - log_q);
                            }
                            if (kl < 0.0) kl = 0.0;
                            self->kld_vals[p_i] = kl;
                            self->sum_kld += kl;
                            if (p_i + 1 < self->n_pos) {
                                const int32_t nxt = self->prompt_ids[p_i + 1];
                                if (nxt >= 0 && static_cast<size_t>(nxt) < kV) {
                                    self->sum_nll_ref += -(static_cast<double>(self->ref_row[nxt]) - log_zr);
                                    self->sum_nll_cand += -(static_cast<double>(c_row[nxt]) - log_zc);
                                    self->nll_cnt += 1;
                                }
                            }
                        }
                        return true;
                    }
                } kctx{fd, prompt_len, prompt_ids_buf, ref_row, kld_vals};

                if (!engine.eval_logits_streaming(prompt_ids_buf, prompt_len, chunk_buf, static_cast<size_t>(squid::MTP_BLOCK_SIZE) * kVocab, KldCtx::on_chunk, &kctx) || kctx.eof_err) {
                    exit_code = 1;
                } else {
                    std::sort(kld_vals, kld_vals + prompt_len);
                    const double mean_kld = kctx.sum_kld / static_cast<double>(prompt_len);
                    const double med_kld = kld_vals[prompt_len / 2];
                    const double p95_kld = kld_vals[(prompt_len * 95) / 100];
                    const double max_kld = kld_vals[prompt_len - 1];
                    const double ppl_ref = std::exp(kctx.sum_nll_ref / static_cast<double>(std::max<size_t>(kctx.nll_cnt, 1)));
                    const double ppl_cand = std::exp(kctx.sum_nll_cand / static_cast<double>(std::max<size_t>(kctx.nll_cnt, 1)));
                    const double top1_pct = 100.0 * static_cast<double>(kctx.top1_match) / static_cast<double>(prompt_len);
                    printf("[kld] mode=%s kv=%s pos=%zu | mean_KLD=%.6f med=%.6f p95=%.6f max=%.6f | top1=%.2f%% | ref_PPL=%.4f cand_PPL=%.4f (dPPL=%.4f)\n",
                           squid::quant_mode_name(quant_mode), squid::kv_format_name(kv_format), prompt_len,
                           mean_kld, med_kld, p95_kld, max_kld, top1_pct, ppl_ref, ppl_cand, ppl_cand - ppl_ref);
                    std::fflush(stdout);
                }
                squid::free_host(kld_vals);
                squid::free_host(ref_row);
                squid::free_host(chunk_buf);
                ::close(fd);
            }
        }
    } else if (command == "longctx") {
        static constexpr uint32_t kTestPositions[] = {1024, 16384, 32768, 65536, 131072, 261888, 262144, 393216, 524288, 786432, 1048000};
        const double total_vram_gib = static_cast<double>(model.arena.capacity_bytes + engine.arena.capacity_bytes) / (1024.0 * 1024.0 * 1024.0);
        printf("=== LONG-CONTEXT SCALING (cpp, mode=%s, kv=%s, max_ctx=%u, total_VRAM=%.2f GiB) ===\n",
               squid::quant_mode_name(quant_mode), squid::kv_format_name(kv_format), ctx, total_vram_gib);
        printf("  %9s | %11s | %11s | %11s | %13s\n", "pos", "B=1 ms/tok", "B=1 tok/s", "B=4 ms/step", "B=4 @dl=2.5");
        for (uint32_t pos_val : kTestPositions) {
            if (pos_val + 256 > ctx) continue;
            if (pos_val == 261888 && ctx > 262144 + 256) continue;
            const double ms1 = engine.bench_step_at_pos(static_cast<int32_t>(pos_val), 0, 24);
            const double ms4 = engine.bench_step_at_pos(static_cast<int32_t>(pos_val), 3, 24);
            printf("  %9u | %8.2f ms | %7.1f t/s | %8.2f ms | %9.1f t/s\n",
                   pos_val, ms1, 1000.0 / ms1, ms4, 2500.0 / ms4);
            std::fflush(stdout);
        }
    } else if (command == "generate") {
        const size_t out_cap = std::max<size_t>(std::min(max_tokens, ctx), 64);
        int32_t* out_tokens_buf = squid::alloc_host_zeroed<int32_t>(out_cap);
        float* d_vis_embeds = nullptr;
        squid::Engine::VisionInput vis_in{};
        if (cli_image_count > 0) {
            size_t total_vis_toks = 0;
            d_vis_embeds = engine.encode_images(cli_images, cli_image_count, &total_vis_toks);
            if (!d_vis_embeds) {
                fprintf(stderr, "Failed to encode vision inputs (checkpoint missing vision weights?)\n");
                exit_code = 1;
            } else {
                vis_in = {d_vis_embeds, total_vis_toks};
            }
        }
        if (exit_code == 0) {
            const squid::GenerateStats st = engine.generate(
                prompt_ids_buf, prompt_len, max_tokens, mtp, out_tokens_buf, out_cap, nullptr, nullptr, vis_in);
            const double dec_toks = static_cast<double>(st.generated_tokens > 0 ? st.generated_tokens - 1 : 0);
            const double tps = (st.decode_ms > 0.0) ? (dec_toks * 1000.0 / st.decode_ms) : 0.0;
            const double acc_pct = (st.total_drafts > 0) ? (100.0 * static_cast<double>(st.total_accepted) / st.total_drafts) : 0.0;
            const double dl = (st.spec_steps > 0) ? (dec_toks / st.spec_steps) : 1.0;

            printf("[gen] prompt=%u gen=%u prefill=%.2fms decode=%.2fms speed=%.2f tok/s | mtp=%u acc=%.1f%% dl=%.2f lookup=%u/%u\n",
                   st.prompt_tokens, st.generated_tokens, st.prefill_ms, st.decode_ms, tps, mtp, acc_pct, dl, st.lookup_steps, st.spec_steps);
            std::fflush(stdout);
            print_token_csv("TOKENS:", out_tokens_buf, st.generated_tokens);
            if (need_tokenizer) print_decoded(tokenizer, out_tokens_buf, st.generated_tokens);
        }
        if (d_vis_embeds) CUDA_CHECK(cudaFree(d_vis_embeds));
        squid::free_host(out_tokens_buf);
    } else if (command == "bench") {
        const size_t out_cap = std::max<size_t>(std::min(max_tokens, ctx), 64);
        int32_t* out_tokens_buf = squid::alloc_host_zeroed<int32_t>(out_cap);
        int32_t* sample_tokens = squid::alloc_host_zeroed<int32_t>(out_cap);
        const size_t clamped_runs = std::min<size_t>(std::max<uint32_t>(runs, 1), kBenchRunsMax);

        // Warmup run (also captures CUDA Graph)
        (void)engine.generate(prompt_ids_buf, prompt_len, std::min<uint32_t>(32, max_tokens), mtp, out_tokens_buf, out_cap);

        double speeds[kBenchRunsMax]{};
        double last_acc = 0.0, last_dl = 1.0, last_prefill = 0.0;
        size_t sample_len = 0;

        for (size_t r = 0; r < clamped_runs; ++r) {
            if (temperature > 0.0f && seed == 0) {
                engine.sample_params.seed = squid::hash_pair_u64(squid::clock_monotonic_ns() ^ static_cast<uint64_t>(r));
            }
            const squid::GenerateStats st = engine.generate(prompt_ids_buf, prompt_len, max_tokens, mtp, out_tokens_buf, out_cap);
            const double dec_toks = static_cast<double>(st.generated_tokens > 0 ? st.generated_tokens - 1 : 0);
            const double tps = (st.decode_ms > 0.0) ? (dec_toks * 1000.0 / st.decode_ms) : 0.0;
            const double acc = (st.total_drafts > 0) ? (100.0 * static_cast<double>(st.total_accepted) / st.total_drafts) : 0.0;
            const double dl = (st.spec_steps > 0) ? (dec_toks / st.spec_steps) : 1.0;
            speeds[r] = tps;
            last_acc = acc;
            last_dl = dl;
            last_prefill = st.prefill_ms;
            if (r == 0) {
                sample_len = st.generated_tokens;
                std::memcpy(sample_tokens, out_tokens_buf, sample_len * sizeof(int32_t));
            }
            printf("  run %zu/%zu: gen=%u decode=%.2fms speed=%.2f tok/s (acc=%.1f%% dl=%.2f)\n",
                   r + 1, clamped_runs, st.generated_tokens, st.decode_ms, tps, acc, dl);
            std::fflush(stdout);
        }

        std::sort(speeds, speeds + clamped_runs);
        const double median = (clamped_runs % 2 == 1)
                                  ? speeds[clamped_runs / 2]
                                  : 0.5 * (speeds[clamped_runs / 2 - 1] + speeds[clamped_runs / 2]);
        printf("MEDIAN: %.2f tok/s | min=%.2f max=%.2f | prefill=%.2fms | acc=%.1f%% dl=%.2f | mode=%s kv=%s temp=%.2f mtp=%u graph=%u\n",
               median, speeds[0], speeds[clamped_runs - 1], last_prefill, last_acc, last_dl,
               squid::quant_mode_name(quant_mode), squid::kv_format_name(kv_format),
               temperature, mtp, use_graphs ? 1u : 0u);
        std::fflush(stdout);
        print_token_csv("TOKENS:", sample_tokens, sample_len);
        if (need_tokenizer) print_decoded(tokenizer, sample_tokens, sample_len);
        squid::free_host(sample_tokens);
        squid::free_host(out_tokens_buf);
    } else {
        fprintf(stderr, "Unknown command '%.*s'\n", static_cast<int>(command.size()), command.data());
        exit_code = 1;
    }

    for (size_t ii = 0; ii < cli_image_count; ++ii) cli_images[ii].deinit();
    engine.deinit();
    model.deinit();
    squid::free_host(prompt_ids_buf);
    if (need_tokenizer) tokenizer.deinit();
    squid::free_host(prompt_file_buf);
    return exit_code;
}
