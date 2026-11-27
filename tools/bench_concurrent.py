#!/usr/bin/env python3
"""Concurrent throughput & parity benchmark for squidDGX HTTP server.

Measures aggregate decode tok/s, per-request tok/s, TTFT, and wall-clock throughput
across concurrency levels C = 1, 2, 4, 8 (and optional long-context prompts).
"""
import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request

PROMPTS = [
    "Write a complete Python implementation of a thread-safe LRU cache with TTL expiration and unit tests.",
    "Explain the mathematical derivation of FlashAttention-2 and why tiling reduces HBM traffic compared to standard attention.",
    "Write a Rust lock-free ring buffer (SPSC) using atomic Acquire/Release ordering with detailed comments on memory fences.",
    "Design a distributed rate limiter using Redis and sliding window logs, comparing trade-offs against token bucket.",
    "Implement an AVL tree in C++20 with insert, delete, rotations, and height-balance invariants verification.",
    "Write a detailed technical comparison of NVFP4, MXFP8, and INT4 weight quantization for MoE inference on Blackwell GPUs.",
    "Implement a minimal SQLite-like B+ tree page manager in Python with split, merge, and range scan operations.",
    "Explain how continuous batching and paged KV caching interact with speculative decoding in LLM serving engines.",
]


def run_one(url: str, prompt: str, max_tokens: int, temperature: float, seed: int, stream: bool, mtp: int | None):
    payload = {
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if mtp is not None:
        payload["mtp"] = mtp

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    chunks_text = []
    usage = {}
    timings = {}

    with urllib.request.urlopen(req, timeout=300) as resp:
        if not stream:
            body = json.loads(resp.read().decode("utf-8"))
            t_end = time.perf_counter()
            usage = body.get("usage", {})
            timings = body.get("timings", {})
            msg = body["choices"][0]["message"]
            txt = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
            return {
                "wall_s": t_end - t0,
                "ttft_s": t_end - t0,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "timings": timings,
                "text": txt,
            }

        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            val = line[5:].strip()
            if val == "[DONE]":
                break
            obj = json.loads(val)
            if "usage" in obj and obj["usage"]:
                usage = obj["usage"]
            if "timings" in obj and obj["timings"]:
                timings = obj["timings"]
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks_text.append(piece)

    t_end = time.perf_counter()
    return {
        "wall_s": t_end - t0,
        "ttft_s": ttft if ttft is not None else (t_end - t0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "timings": timings,
        "text": "".join(chunks_text),
    }


def wait_health(url: str, timeout_s: float = 120.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=2) as r:
                if r.status == 200:
                    return json.loads(r.read().decode("utf-8"))
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Server at {url} not healthy after {timeout_s}s")


def bench_level(url: str, concurrency: int, max_tokens: int, temperature: float, stream: bool, mtp: int | None, pad_repeat: int):
    prompts = []
    for i in range(concurrency):
        base = PROMPTS[i % len(PROMPTS)]
        if pad_repeat > 0:
            filler = " ".join([f"[ctx{i}-{k}: background technical context block for request {i}]" for k in range(pad_repeat)])
            base = filler + "\n\n" + base
        prompts.append(base)

    t_start = time.perf_counter()
    results = [None] * concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {
            ex.submit(run_one, url, prompts[i], max_tokens, temperature, 1000 + i, stream, mtp): i
            for i in range(concurrency)
        }
        for f in concurrent.futures.as_completed(futs):
            idx = futs[f]
            results[idx] = f.result()
    wall_s = time.perf_counter() - t_start

    tot_gen = sum(r["completion_tokens"] for r in results)
    tot_prompt = sum(r["prompt_tokens"] for r in results)
    agg_tok_s = tot_gen / wall_s if wall_s > 0 else 0.0
    avg_ttft_ms = 1000.0 * sum(r["ttft_s"] for r in results) / concurrency
    max_ttft_ms = 1000.0 * max(r["ttft_s"] for r in results)
    avg_user_tok_s = sum(
        (r["completion_tokens"] / r["wall_s"]) if r["wall_s"] > 0 else 0.0 for r in results
    ) / concurrency
    avg_pf_ms = sum(r["timings"].get("prefill_ms", 0.0) for r in results) / concurrency
    avg_dec_ms = sum(r["timings"].get("decode_ms", 0.0) for r in results) / concurrency
    avg_acc = sum(r["timings"].get("spec_acc_pct", 0.0) for r in results) / concurrency
    avg_dl = sum(r["timings"].get("spec_dl", 1.0) for r in results) / concurrency
    return {
        "C": concurrency,
        "wall_s": round(wall_s, 3),
        "tot_prompt": tot_prompt,
        "tot_gen": tot_gen,
        "agg_tok_s": round(agg_tok_s, 2),
        "avg_user_tok_s": round(avg_user_tok_s, 2),
        "avg_ttft_ms": round(avg_ttft_ms, 1),
        "max_ttft_ms": round(max_ttft_ms, 1),
        "avg_pf_ms": round(avg_pf_ms, 1),
        "avg_dec_ms": round(avg_dec_ms, 1),
        "avg_acc": round(avg_acc, 1),
        "avg_dl": round(avg_dl, 2),
        "texts": [r["text"] for r in results],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--stream", action="store_true", default=True)
    ap.add_argument("--no-stream", dest="stream", action="store_false")
    ap.add_argument("--mtp", type=int, default=None)
    ap.add_argument("--pad-repeat", type=int, default=0, help="Add ~10*N unique context tokens per request")
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--check-parity", action="store_true", help="Verify C>1 greedy outputs match C=1 greedy outputs")
    args = ap.parse_args()

    health = wait_health(args.url)
    print(f"# health: {json.dumps(health)}", flush=True)

    if args.warmup:
        run_one(args.url, "Say hello in one sentence.", 16, 0.0, 42, True, args.mtp)

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    rows = []
    for c in levels:
        res = bench_level(args.url, c, args.max_tokens, args.temp, args.stream, args.mtp, args.pad_repeat)
        rows.append(res)
        print(
            f"C={res['C']} | agg={res['agg_tok_s']:6.2f} tok/s | user={res['avg_user_tok_s']:5.2f} tok/s | "
            f"ttft_avg={res['avg_ttft_ms']:6.1f}ms ttft_max={res['max_ttft_ms']:6.1f}ms | "
            f"wall={res['wall_s']:.3f}s | pf={res['avg_pf_ms']:.0f}ms acc={res['avg_acc']:4.1f}% dl={res['avg_dl']:.2f} | "
            f"gen={res['tot_gen']} prompt={res['tot_prompt']}",
            flush=True,
        )

    if args.check_parity and args.temp == 0.0:
        # Run each prompt sequentially at C=1 and compare to C=4 (or max C <= 4)
        ref_c = min(4, max(levels))
        ref_row = next((r for r in rows if r["C"] == ref_c), None)
        if ref_row is not None:
            mismatches = 0
            for i in range(ref_c):
                base = PROMPTS[i % len(PROMPTS)]
                if args.pad_repeat > 0:
                    filler = " ".join([f"[ctx{i}-{k}: background technical context block for request {i}]" for k in range(args.pad_repeat)])
                    base = filler + "\n\n" + base
                single = run_one(args.url, base, args.max_tokens, 0.0, 1000 + i, args.stream, args.mtp)
                batched_txt = ref_row["texts"][i]
                if single["text"] != batched_txt:
                    # Check prefix match length
                    common = 0
                    for a, b in zip(single["text"], batched_txt):
                        if a != b:
                            break
                        common += 1
                    print(
                        f"PARITY DIFF slot {i}: common={common}/{min(len(single['text']), len(batched_txt))} chars "
                        f"(single_len={len(single['text'])}, batch_len={len(batched_txt)})",
                        flush=True,
                    )
                    mismatches += 1
            print(f"PARITY SUMMARY (C={ref_c} vs C=1): {ref_c - mismatches}/{ref_c} exact text matches", flush=True)


if __name__ == "__main__":
    main()
