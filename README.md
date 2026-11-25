# squidDGX

A Linux ARM64 CUDA inference engine and OpenAI-compatible server for
Inkling-2-Mini on NVIDIA DGX Spark (GB10, `sm_121a`). It serves one generation
at a time and defaults to NVFP4 weights. DSpark speculative decoding can
verify several draft tokens per target-model pass.

## Build and run on Spark

Build in the checkout on a Spark with Docker and NVIDIA Container Toolkit.
The image builds pure C++20 (`g++`) with CUDA 13.0.1.

```sh
docker build -t squiddgx .
```

Set `MODEL_DIR` to an absolute path containing the original checkpoint and
tokenizer. The engine packs the weights to its NVFP4 runtime layout during
startup. Include `mtp.safetensors` to use DSpark.
[Thinking Machines Inkling-2-Mini checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini)

```sh
docker run --rm --gpus all --memory=96000000000 --memory-swap=96000000000 \
  --mount "type=bind,source=$MODEL_DIR,target=/model,readonly" \
  -p 127.0.0.1:8000:8000 squiddgx serve /model
```

Serving defaults to 65,536 context tokens. Generation and benchmarks default
to adaptive `--mtp 7`. Add `--kv q4` to reduce KV traffic and footprint on long prompts. The optional
`--mode exact --kv bf16` profile supports reference comparisons via
`logits` and `kld`.

The Docker options set a 96 GB cgroup memory limit with zero additional
swap allowance for the container. GB10 shares DRAM between CUDA and host
allocations. The engine checks both the 96 GB budget and host
`MemAvailable` before allocating CUDA arenas, with 4 GiB reserved for
CUDA and host overhead. Check CUDA process memory alongside host
`MemAvailable` when measuring peak usage.

## Measure decode throughput

```sh
docker run --rm --gpus all --memory=96000000000 --memory-swap=96000000000 \
  --mount "type=bind,source=$MODEL_DIR,target=/model,readonly" squiddgx \
  bench /model --prompt "Write a Python LRU cache with tests." \
  --ctx 8192 --kv fp8 --mtp 7 --temp 0 --max-tokens 192 --runs 5
```

The reported median excludes model load, graph warmup, and prefill. Use the
same prompt, context, settings, and host load for comparisons. Set
`--mtp 0` for single-token decoding and record the number of output tokens
and accepted drafts when comparing speculative configurations.

On `spark-a.lan`, the command above had a median of **78.99 tok/s**,
up from **77.20 tok/s** on the ARM64 baseline before the decode changes.
All 192 greedy output tokens matched. The short-context run peaked at
15.84 GiB of CUDA process memory. A seeded HTTP chat run on the same
prompt (`temperature=1`, `top_p=0.95`, `seed=42`, `mtp=7`) reached
**88.48 tok/s** across three timed requests, up from **80.61 tok/s**
before the confidence gate.

An optional `draft_vocab_32k.bin` or `draft_vocab_49k.bin` in the model
directory limits DSpark to a workload-specific shortlist of little-endian
32-bit token IDs. Use `SQUID_DRAFT_VOCAB=full` to compare against the full
draft vocabulary. The target model still verifies every accepted token.

The `longctx` command measures synthetic step latency with preallocated KV
caches. At a 1,048,576-token capacity with BF16 KV, Spark A reported a
47.94 GiB peak in CUDA process memory and a 48.93 GiB peak drop in host
`MemAvailable`. A separate run with Q4 KV and the checked-in corpus as a
real prompt decoded at a median of 69.66 tok/s.

## CPU tokenizer timings

Five interleaved runs on one Spark A performance core produced identical
token IDs on all six inputs. Median encoding time on 323K source tokens
fell from 10.35 ms to 7.82 ms. A repeated multilingual input fell from
9.32 ms to 3.76 ms, and a mixed Unicode/hex input fell from 9.61 ms to
4.90 ms. The nine-token cold input took 6.4 µs after tuning, up from 3.0 µs.

The repeated-prompt decode benchmark exercised prompt lookup on all 12
verification steps and matched all 96 baseline token IDs. Median decode
was 212.79 tok/s across seven runs.

## NEON and sampling timings

On Spark A, NEON widening cut the 323K-token source input from 7.81 ms
to 6.71 ms. All six tokenizer inputs still matched. On a 64 KiB plain
JSON string, serialization fell from 50.5 µs to 2.33 µs and parsing from
33.7 µs to 2.09 µs. A 64-byte string with an escape every third byte
parsed in 60.6 ns, up from 40.9 ns.

On Spark B, packing histogram mass and count into one shared-memory
atomic per entry raised 256-token seeded decode from 95.06 to 95.85 tok/s
for top-p sampling and from 78.27 to 78.76 tok/s with top-k 40. Greedy
decode measured 88.26 tok/s versus 88.62 tok/s. Completion payloads and
streamed deltas matched the baseline after normalizing request metadata.

See [NOTICE](NOTICE) for upstream attribution and [LICENSE](LICENSE) for
licensing.
