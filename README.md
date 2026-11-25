# squidDGX

A pure C++20 + CUDA inference engine and OpenAI-compatible HTTP/SSE server for
Inkling-2-Mini on NVIDIA DGX Spark (GB10, `sm_121a`, Linux ARM64). It serves one
request at a time, defaults to full checkpoint precision (`--mode exact --kv bf16`)
with a 262,144-token context window and 65,536-token output cap, and supports
DSpark adaptive speculative decoding (`--mtp 7`).

## Build and run

Build inside the checkout with Docker and NVIDIA Container Toolkit (pure C++20
`g++` + CUDA 13.0.1):

```sh
docker build -t squiddgx .
```

Set `MODEL_DIR` to an absolute path containing the
[Inkling-2-Mini checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
and tokenizer. Include `mtp.safetensors` in `MODEL_DIR` to enable DSpark
speculative decoding.

```sh
docker run --rm --gpus all --memory=96000000000 --memory-swap=96000000000 \
  --mount "type=bind,source=$MODEL_DIR,target=/model,readonly" \
  -p 127.0.0.1:8000:8000 squiddgx serve /model
```

### Defaults and options

- **Precision**: `--mode exact --kv bf16` (BF16 dense/attention/DSpark weights,
  native checkpoint MoE weights, BF16 KV cache). Optional quantized profiles:
  `--mode fp8`, `--mode nvfp4`, `--kv fp8`, `--kv q4`.
- **Context & output limits**: `--ctx 262144` (262K context) and
  `--max-tokens 65536` (64K output cap for `serve` and `generate`).
- **Speculative decoding**: adaptive `--mtp 7` (`--spec-threshold 0.75`). Pass
  `--mtp 0` for single-token decoding. An optional `draft_vocab_32k.bin` or
  `draft_vocab_49k.bin` in `MODEL_DIR` restricts DSpark to a shortlist; set
  `SQUID_DRAFT_VOCAB=full` to force the full draft vocabulary.
- **Memory budget**: GB10 shares unified DRAM between CUDA and host allocations.
  The Docker flags above enforce a 96 GB cgroup limit with zero swap, and the
  engine verifies available memory before allocating its single-arena CUDA
  weight and runtime buffers (reserving 4 GiB for CUDA/host overhead).

## Benchmarks

Measured on NVIDIA DGX Spark (GB10, `sm_121a`) with the default configuration
(`--mode exact --kv bf16 --ctx 262144`, total VRAM **27.92 GiB** with `MTP=on`:
18.28 GiB weights + 9.64 GiB runtime; model load **2.24 s**).

### Prefill

Command (`tools/kld_corpus.txt`, 7,797 tokens):

```sh
docker run --rm --gpus all --memory=96000000000 --memory-swap=96000000000 \
  --mount "type=bind,source=$MODEL_DIR,target=/model,readonly" \
  --mount "type=bind,source=$PWD/tools,target=/tools,readonly" \
  squiddgx generate /model --prompt-file /tools/kld_corpus.txt --max-tokens 32
```

| Workload | Tokens | Latency | Throughput |
|---|---:|---:|---:|
| Full corpus prefill (`kld_corpus.txt`) | 7,797 | 2,776.47 ms | **2,808 tok/s** |
| Peak chunk (`B=2048`) | 2,048 | 697.27 ms | **2,937 tok/s** |

### Decode

Command (median of 5 timed runs, excluding model load, warmup, and prefill):

```sh
docker run --rm --gpus all --memory=96000000000 --memory-swap=96000000000 \
  --mount "type=bind,source=$MODEL_DIR,target=/model,readonly" \
  squiddgx bench /model --prompt "Write a Python LRU cache with tests." \
  --max-tokens 192 --runs 5
```

| Profile | Decoding | Output | Median | Notes |
|---|---|---:|---:|---|
| Default (`exact`, `bf16` KV, `ctx=262144`) | Greedy, `mtp=7` | 192 tok | **43.59 tok/s** | `acc=67.2%`, `dl=1.69` |
| Default (`exact`, `bf16` KV, `ctx=262144`) | Stochastic (`temp=1.0`, `top_p=0.95`, `seed=42`), `mtp=7` | 192 tok | **47.48 tok/s** | `acc=55.7%`, `dl=2.08` |
| Default (`exact`, `bf16` KV, `ctx=262144`) | Greedy, `mtp=0` (`B=1`) | 128 tok | **37.78 tok/s** | `26.47 ms/tok`, 26.14 GiB VRAM |

### Long-context step scaling (`longctx`, `ctx=262144`)

```sh
docker run --rm --gpus all --memory=96000000000 --memory-swap=96000000000 \
  --mount "type=bind,source=$MODEL_DIR,target=/model,readonly" \
  squiddgx longctx /model
```

| Position | `B=1` Latency | `B=1` Throughput | `B=4` Step | `B=4` (`@dl=2.5`) |
|---:|---:|---:|---:|---:|
| 1,024 | 27.62 ms | 36.2 tok/s | 48.68 ms | 51.4 tok/s |
| 16,384 | 29.85 ms | 33.5 tok/s | 50.57 ms | 49.4 tok/s |
| 32,768 | 32.42 ms | 30.8 tok/s | 54.79 ms | 45.6 tok/s |
| 65,536 | 37.47 ms | 26.7 tok/s | 59.80 ms | 41.8 tok/s |
| 131,072 | 47.40 ms | 21.1 tok/s | 72.08 ms | 34.7 tok/s |
| 261,888 | 67.19 ms | 14.9 tok/s | 93.51 ms | 26.7 tok/s |

See [NOTICE](NOTICE) for upstream attribution and [LICENSE](LICENSE) for
licensing.
