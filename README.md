# squidDGX

A pure C++20 + CUDA inference engine and OpenAI-compatible HTTP/SSE server for
Inkling-2-Mini on NVIDIA DGX Spark (GB10, `sm_121a`, Linux ARM64) and discrete
NVIDIA GPUs (A10G, L40S, H100, RTX PRO 4500/6000 Blackwell; see
[Supported GPUs](#supported-gpus)). It serves one
request at a time, defaults to NVFP4 weights with a BF16 KV cache
(`--mode nvfp4 --kv bf16`), a 262,144-token context window, and a 65,536-token
output cap, and supports DSpark adaptive speculative decoding (`--mtp 7`).

## Build and run

Build inside the checkout with Docker and NVIDIA Container Toolkit (pure C++20
`g++` + CUDA 13.0.1):

```sh
docker build -t squiddgx .
```

Set `MODEL_DIR` to an absolute path containing the
[Inkling-2-Mini checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
and tokenizer. The engine packs weights to its NVFP4 runtime layout during
startup. Include `mtp.safetensors` in `MODEL_DIR` to enable DSpark speculative
decoding.

```sh
docker run --rm --gpus all --memory=96000000000 --memory-swap=96000000000 \
  --mount "type=bind,source=$MODEL_DIR,target=/model,readonly" \
  -p 127.0.0.1:8000:8000 squiddgx serve /model
```

### Defaults and options

- **Precision**: `--mode nvfp4 --kv bf16` (NVFP4 weights, full-precision BF16 KV
  cache). Optional profiles: `--mode fp8`, `--mode exact` (BF16 dense + native
  checkpoint MoE weights), `--kv fp8`, `--kv q4`.
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

### Supported GPUs

The default image contains kernels for all of these. Build a single target with
`docker build --build-arg CUDA_ARCHS=90 -t squiddgx .` for a faster compile.

| GPU | `CUDA_ARCHS` | Memory | Default context (`serve`, `--kv bf16`) |
|---|---|---:|---|
| DGX Spark (GB10) | `121a` | 128 GB unified | 262K |
| RTX PRO 6000 Blackwell | `120a` | 96 GB | 262K |
| RTX PRO 4500 Blackwell | `120a` | 32 GB | 262K |
| H100 | `90` | 80 GB | 262K |
| L40S | `89` | 48 GB | 262K |
| A10G | `86` | 24 GB | ~100K (auto-reduced) |

- Blackwell decodes NVFP4 weights in hardware. Ampere/Ada/Hopper use a
  bit-exact software decode, which is slower (~40% lower single-stream decode
  on the same GPU). Hopper and Blackwell also use programmatic dependent launch.
- T4 (and anything below compute capability 8.0) is not supported. The NVFP4
  weights alone (~15 GiB with MTP and vision) fill a 16 GB T4, and the kernels
  need BF16 tensor cores and `cp.async`.
- On discrete GPUs, the default `--ctx 262144` is shrunk to the largest
  context that fits in free VRAM, keeping 1 GiB in reserve
  (`SQUID_VRAM_RESERVE_MIB` overrides this). An explicit `--ctx` is never
  shrunk; it fails with the memory shortfall instead. `--kv fp8` or `--kv q4`
  fits roughly 2x or 4x more context.
- CUDA 13 needs NVIDIA driver 580 or newer on the host.
- The 96 GB `--memory` flags above are only for the Spark's unified memory. On
  a discrete GPU, run:

```sh
docker run --rm --gpus all \
  --mount "type=bind,source=$MODEL_DIR,target=/model,readonly" \
  -p 127.0.0.1:8000:8000 squiddgx serve /model
```

## Benchmarks

Measured on NVIDIA DGX Spark (GB10, `sm_121a`) at `--ctx 262144 --kv bf16`
(model load ~2.1–2.4 s).

### Peak memory

Measured during full 7,797-token prefill + decode (`mtp=7`) and single-token
decode (`mtp=0`) inside a 96 GB Docker container. On GB10 unified memory, host
`MemAvailable` drop reflects total physical DRAM used by CUDA arenas, driver
overhead, and host buffers combined.

| Mode (`kv=bf16`, `ctx=262144`) | Speculative | Weights VRAM | Runtime VRAM | Total CUDA VRAM | Cgroup `memory.peak` | Peak Host `MemAvailable` Drop |
|---|---|---:|---:|---:|---:|---:|
| **`nvfp4` (default)** | `mtp=7` | 14.55 GiB | 9.64 GiB | **24.19 GiB** | 0.80 GiB | **25.36 GiB** |
| **`nvfp4` (default)** | `mtp=0` | 14.28 GiB | 8.55 GiB | **22.83 GiB** | 0.43 GiB | **23.68 GiB** |
| `exact` | `mtp=7` | 18.28 GiB | 9.64 GiB | **27.92 GiB** | 0.81 GiB | **28.97 GiB** |
| `exact` | `mtp=0` | 17.59 GiB | 8.55 GiB | **26.14 GiB** | 0.44 GiB | **26.85 GiB** |

### Prefill

Command (`tools/kld_corpus.txt`, 7,797 tokens):

```sh
docker run --rm --gpus all --memory=96000000000 --memory-swap=96000000000 \
  --mount "type=bind,source=$MODEL_DIR,target=/model,readonly" \
  --mount "type=bind,source=$PWD/tools,target=/tools,readonly" \
  squiddgx generate /model --prompt-file /tools/kld_corpus.txt --max-tokens 32
```

| Mode (`kv=bf16`, `ctx=262144`) | Workload | Tokens | Latency | Throughput |
|---|---|---:|---:|---:|
| **`nvfp4` (default)** | Full corpus (`kld_corpus.txt`) | 7,797 | 2,679.52 ms | **2,910 tok/s** |
| **`nvfp4` (default)** | Peak chunk (`B=2048`) | 2,048 | 677.25 ms | **3,024 tok/s** |
| `exact` | Full corpus (`kld_corpus.txt`) | 7,797 | 2,776.47 ms | **2,808 tok/s** |
| `exact` | Peak chunk (`B=2048`) | 2,048 | 697.27 ms | **2,937 tok/s** |

### Decode

Command (median of 5 timed runs, excluding model load, warmup, and prefill):

```sh
docker run --rm --gpus all --memory=96000000000 --memory-swap=96000000000 \
  --mount "type=bind,source=$MODEL_DIR,target=/model,readonly" \
  squiddgx bench /model --prompt "Write a Python LRU cache with tests." \
  --max-tokens 192 --runs 5
```

| Mode (`kv=bf16`, `ctx=262144`) | Decoding | Output | Median | Notes |
|---|---|---:|---:|---|
| **`nvfp4` (default)** | Greedy, `mtp=7` | 192 tok | **82.87 tok/s** | `acc=63.9%`, `dl=1.68` |
| **`nvfp4` (default)** | Stochastic (`temp=1.0`, `top_p=0.95`, `seed=42`), `mtp=7` | 192 tok | **84.11 tok/s** | `acc=62.4%`, `dl=1.85` |
| **`nvfp4` (default)** | Greedy, `mtp=0` (`B=1`) | 128 tok | **73.35 tok/s** | `13.63 ms/tok`, 22.83 GiB VRAM |
| `exact` | Greedy, `mtp=7` | 192 tok | **43.59 tok/s** | `acc=67.2%`, `dl=1.69` |
| `exact` | Stochastic (`temp=1.0`, `top_p=0.95`, `seed=42`), `mtp=7` | 192 tok | **47.48 tok/s** | `acc=55.7%`, `dl=2.08` |
| `exact` | Greedy, `mtp=0` (`B=1`) | 128 tok | **37.78 tok/s** | `26.47 ms/tok`, 26.14 GiB VRAM |

### Long-context step scaling (`longctx`, `ctx=262144`)

```sh
docker run --rm --gpus all --memory=96000000000 --memory-swap=96000000000 \
  --mount "type=bind,source=$MODEL_DIR,target=/model,readonly" \
  squiddgx longctx /model
```

| Position | `nvfp4` `B=1` | `nvfp4` `B=4` (`@dl=2.5`) | `exact` `B=1` | `exact` `B=4` (`@dl=2.5`) |
|---:|---:|---:|---:|---:|
| 1,024 | 14.48 ms (**69.1 tok/s**) | 28.68 ms (**87.2 tok/s**) | 27.62 ms (36.2 tok/s) | 48.68 ms (51.4 tok/s) |
| 16,384 | 16.73 ms (**59.8 tok/s**) | 31.63 ms (**79.0 tok/s**) | 29.85 ms (33.5 tok/s) | 50.57 ms (49.4 tok/s) |
| 32,768 | 19.15 ms (**52.2 tok/s**) | 33.64 ms (**74.3 tok/s**) | 32.42 ms (30.8 tok/s) | 54.79 ms (45.6 tok/s) |
| 65,536 | 24.04 ms (**41.6 tok/s**) | 37.75 ms (**66.2 tok/s**) | 37.47 ms (26.7 tok/s) | 59.80 ms (41.8 tok/s) |
| 131,072 | 33.97 ms (**29.4 tok/s**) | 48.45 ms (**51.6 tok/s**) | 47.40 ms (21.1 tok/s) | 72.08 ms (34.7 tok/s) |
| 261,888 | 53.58 ms (**18.7 tok/s**) | 71.44 ms (**35.0 tok/s**) | 67.19 ms (14.9 tok/s) | 93.51 ms (26.7 tok/s) |

See [NOTICE](NOTICE) for upstream attribution and [LICENSE](LICENSE) for
licensing.
