"""Benchmark harness.

Method (so the numbers mean something):
* Compile time is measured separately from steady-state time (first call vs later calls).
* Every timed call ends with block_until_ready(), because JAX dispatch is asynchronous.
* We report the median and the minimum of N repeats, never a single run.
* XLA's own cost model (flops, bytes accessed) is reported next to wall-clock time.

Usage: python bench/run_bench.py [--out results/cpu_results.md] [--repeats 15]
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxlab.attention import full_window_attention, local_attention
from jaxlab.streaming import causal_conv_offline, causal_conv_streaming, init_state, stream_step


def timed(fn, *args, repeats):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - t0
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        runs.append(time.perf_counter() - t0)
    return {"first_call_ms": first * 1e3, "median_ms": statistics.median(runs) * 1e3, "min_ms": min(runs) * 1e3}


def cost(fn, *args):
    try:
        c = jax.jit(fn).lower(*args).compile().cost_analysis()
        c = c[0] if isinstance(c, (list, tuple)) else c
        return {"gflops": c.get("flops", float("nan")) / 1e9, "mbytes": c.get("bytes accessed", float("nan")) / 1e6}
    except Exception:  # cost model is best-effort and backend dependent
        return {"gflops": float("nan"), "mbytes": float("nan")}


def hlo_fusions(fn, *args):
    try:
        text = jax.jit(fn).lower(*args).compile().as_text()
        return text.count("fusion("), text
    except Exception:
        return -1, ""


def bench_attention(repeats, out_dir):
    h, d, w = 4, 64, 128
    rows = []
    for t in (512, 1024, 2048, 4096):
        key = jax.random.PRNGKey(t)
        q, k, v = (jax.random.normal(jax.random.fold_in(key, i), (t, h, d), jnp.float32) for i in range(3))
        full = jax.jit(lambda q, k, v: full_window_attention(q, k, v, w))
        local = jax.jit(lambda q, k, v: local_attention(q, k, v, w))
        err = float(jnp.max(jnp.abs(full(q, k, v) - local(q, k, v))))
        row = {"T": t, "max_abs_diff": err}
        for name, f in (("full", full), ("local", local)):
            row[name] = {**timed(f, q, k, v, repeats=repeats), **cost(f, q, k, v)}
        rows.append(row)
        if t == 1024:
            n, text = hlo_fusions(lambda q, k, v: local_attention(q, k, v, w), q, k, v)
            row["local_fusions"] = n
            (out_dir / "local_attention_T1024.hlo.txt").write_text(text, encoding="utf-8")
    return {"heads": h, "head_dim": d, "window": w, "rows": rows}


def bench_streaming(repeats):
    t, c, k = 16384, 64, 5
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (t, c), jnp.float32)
    kernel = jax.random.normal(jax.random.fold_in(key, 1), (k, c, c), jnp.float32) * 0.1
    offline = jax.jit(causal_conv_offline)
    rows = [{"mode": "offline (whole signal)", **timed(offline, x, kernel, repeats=repeats)}]
    ref = offline(x, kernel)
    for chunk in (16, 128, 1024):
        scan_fn = jax.jit(lambda x, kernel, chunk=chunk: causal_conv_streaming(x, kernel, chunk))
        diff = float(jnp.max(jnp.abs(scan_fn(x, kernel) - ref)))
        rows.append({"mode": f"streaming, one scan, chunk={chunk}", "max_abs_diff": diff,
                     **timed(scan_fn, x, kernel, repeats=repeats)})
    # Python-driven loop: one jitted call per chunk (what a naive real-time server does).
    step = jax.jit(stream_step)
    chunk = 128

    def python_loop(x, kernel):
        state, outs = init_state(kernel), []
        for i in range(0, t, chunk):
            state, o = step(state, x[i : i + chunk], kernel)
            outs.append(o)
        return jnp.concatenate(outs)

    rows.append({"mode": f"streaming, Python loop, chunk={chunk}",
                 "max_abs_diff": float(jnp.max(jnp.abs(python_loop(x, kernel) - ref))),
                 **timed(python_loop, x, kernel, repeats=max(3, repeats // 3))})
    return {"frames": t, "channels": c, "kernel": k, "rows": rows}


def render(res, env):
    a, s = res["attention"], res["streaming"]
    L = [f"# Benchmark results ({env['device']})", "",
         f"JAX {env['jax']}, Python {env['python']}, {env['machine']}. Median of repeated runs after warm-up; "
         "first-call time includes tracing and XLA compilation.", "",
         f"## Windowed attention (heads={a['heads']}, head_dim={a['head_dim']}, window={a['window']}, float32)", "",
         "| T | full median ms | local median ms | speed-up | full GFLOP | local GFLOP | full MB accessed | local MB accessed | max abs diff |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in a["rows"]:
        f, l = r["full"], r["local"]
        L.append(f"| {r['T']} | {f['median_ms']:.2f} | {l['median_ms']:.2f} | {f['median_ms'] / l['median_ms']:.1f}x | "
                 f"{f['gflops']:.2f} | {l['gflops']:.2f} | {f['mbytes']:.0f} | {l['mbytes']:.0f} | {r['max_abs_diff']:.1e} |")
    L += ["", f"## Streaming causal conv ({s['frames']} frames, {s['channels']} channels, kernel {s['kernel']})", "",
          "| Mode | median ms | min ms | first call ms | max abs diff vs offline |", "|---|---|---|---|---|"]
    for r in s["rows"]:
        d = f"{r['max_abs_diff']:.1e}" if "max_abs_diff" in r else "-"
        L.append(f"| {r['mode']} | {r['median_ms']:.2f} | {r['min_ms']:.2f} | {r['first_call_ms']:.0f} | {d} |")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/cpu_results.md")
    p.add_argument("--repeats", type=int, default=15)
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {"jax": jax.__version__, "python": platform.python_version(), "machine": platform.processor() or platform.machine(),
           "device": str(jax.devices()[0].device_kind or jax.devices()[0].platform).upper()}
    res = {"attention": bench_attention(args.repeats, out.parent), "streaming": bench_streaming(args.repeats)}
    out.write_text(render(res, env), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"env": env, **res}, indent=2), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
