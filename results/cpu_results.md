# Benchmark results (CPU)

JAX 0.12.0.1, Python 3.12.10, Intel64 Family 6 Model 186 Stepping 2, GenuineIntel. Median of repeated runs after warm-up; first-call time includes tracing and XLA compilation.

## Windowed attention (heads=4, head_dim=64, window=128, float32)

| T | full median ms | local median ms | speed-up | full GFLOP | local GFLOP | full MB accessed | local MB accessed | max abs diff |
|---|---|---|---|---|---|---|---|---|
| 512 | 14.70 | 16.07 | 0.9x | 0.28 | 0.14 | 44 | 28 | 1.5e-07 |
| 1024 | 21.02 | 18.45 | 1.1x | 1.10 | 0.28 | 80 | 35 | 1.5e-07 |
| 2048 | 75.96 | 23.80 | 3.2x | 4.41 | 0.55 | 294 | 71 | 3.3e-07 |
| 4096 | 300.39 | 22.71 | 13.2x | 17.63 | 1.10 | 1124 | 142 | 2.7e-07 |

## Streaming causal conv (16384 frames, 64 channels, kernel 5)

| Mode | median ms | min ms | first call ms | max abs diff vs offline |
|---|---|---|---|---|
| offline (whole signal) | 9.34 | 7.62 | 427 | - |
| streaming, one scan, chunk=16 | 165.14 | 135.68 | 163 | 1.9e-06 |
| streaming, one scan, chunk=128 | 42.83 | 32.09 | 68 | 4.8e-06 |
| streaming, one scan, chunk=1024 | 24.65 | 21.76 | 49 | 1.9e-06 |
| streaming, Python loop, chunk=128 | 50.86 | 47.60 | 95 | 4.8e-06 |
