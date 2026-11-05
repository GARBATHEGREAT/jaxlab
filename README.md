# jaxlab

Three small JAX / XLA / Pallas experiments, each tested against an independent reference and benchmarked with a method that separates compile time from run time.

**Status, stated plainly:** everything here has been run on a **CPU only** (JAX 0.12, Python 3.12). Nothing has been run on a TPU yet, and the Pallas kernel has only been run in `interpret=True` mode. The code is written with TPU constraints in mind (static shapes, 8x128 tiling), but no TPU performance claim is made. Results are in [`results/cpu_results.md`](results/cpu_results.md).

## What is in it
| Module | What it does | Why it matters |
|---|---|---|
| [`jaxlab/attention.py`](jaxlab/attention.py) | Causal sliding-window attention two ways: full T x T masked, and blocked local attention (each block sees itself and the previous block) | Same maths, O(T * window) instead of O(T^2), all shapes static so XLA can compile it once |
| [`jaxlab/streaming.py`](jaxlab/streaming.py) | Causal 1-D convolution computed offline, and chunk by chunk with carried state inside one `lax.scan` | The pattern a real-time streaming model needs: streaming output must equal offline output |
| [`jaxlab/kernels.py`](jaxlab/kernels.py) | A Pallas RMSNorm kernel with a plain-JAX reference | A first hand-written kernel with TPU-style block shapes |

## Correctness (22 tests, `pytest -q`)
- Both attention versions match an **independent NumPy loop** reference (three sequence/window combinations).
- Streaming equals offline for chunk sizes 1, 4, 16, 64 and kernel sizes 1, 3, 5, and is causal (changing future input never changes past output).
- The jitted local attention **does not retrace** when called again with the same shapes.
- The Pallas kernel matches the reference in float32 and bfloat16, and rejects shapes that break TPU tiling.

## Results on CPU (median of 15 runs after warm-up)

**Windowed attention** (heads 4, head dim 64, window 128, float32). Outputs agree to about 3e-7.

| T | full ms | local ms | speed-up | full GFLOP | local GFLOP | full MB | local MB |
|---|---|---|---|---|---|---|---|
| 512 | 14.70 | 16.07 | 0.9x | 0.28 | 0.14 | 44 | 28 |
| 1024 | 21.02 | 18.45 | 1.1x | 1.10 | 0.28 | 80 | 35 |
| 2048 | 75.96 | 23.80 | 3.2x | 4.41 | 0.55 | 294 | 71 |
| 4096 | 300.39 | 22.71 | 13.2x | 17.63 | 1.10 | 1124 | 142 |

**Streaming causal conv** (16384 frames, 64 channels, kernel 5). All modes agree with offline to under 5e-6.

| Mode | median ms |
|---|---|
| offline, whole signal | 9.34 |
| streaming, one scan, chunk 16 | 165.14 |
| streaming, one scan, chunk 128 | 42.83 |
| streaming, one scan, chunk 1024 | 24.65 |
| streaming, Python loop, chunk 128 | 50.86 |

## What I learned (and what surprised me)
1. **Local attention only wins above a crossover.** At T=512 it is slightly slower; the speed-up appears from about T=2048 and reaches 13x at T=4096. XLA's own cost model says the work drops 16x in FLOPs and 8x in bytes at T=4096, but wall-clock gains are smaller than the FLOP ratio, so the CPU is not purely compute-bound here.
2. **Streaming is not free.** Chunk size dominates: chunk 16 inside `lax.scan` is about 18x slower than offline. A single scan is only modestly faster than a Python loop at chunk 128. My hypothesis is per-iteration overhead of the scan on CPU, but I have **not profiled it**, so treat that as unconfirmed. For a real-time system this is a latency/throughput trade-off, not a free win.
3. **Benchmark hygiene matters.** First-call times (up to 427 ms) are compilation, not run time; JAX dispatch is asynchronous, so every timing ends in `block_until_ready`.
4. Timings are from a laptop CPU with other processes running, so small differences (a few ms) are noise. The large effects (13x, 18x) are not.

## Not done yet
- No TPU run. On a TPU VM: install `jax[tpu]`, run `python bench/run_bench.py --out results/tpu_results.md`, and run the kernel with `interpret=False`. I have not verified that the kernel compiles with Mosaic.
- No `jax.profiler` trace, no bfloat16 benchmarks, no sharding (`shard_map` / `pjit`).
- The scan overhead in point 2 is unprofiled.

## Run it
```bash
pip install -e . pytest
pytest -q
python bench/run_bench.py --out results/cpu_results.md
```
Compiled HLO for the local attention at T=1024 is saved in `results/local_attention_T1024.hlo.txt`.

MIT licence.
