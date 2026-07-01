# Benchmark Notes

This document describes the methodology and key findings for each benchmark
in this repository. All benchmarks compare Julia/Reactant against Python/JAX
on a Google Cloud TPU v4 VM running Ubuntu 22.04.5 LTS.

---

## General Methodology

All timings use median wall-clock time over multiple runs (20 runs for small
inputs, 10 for medium, 5 for large), with one warm-up run per kernel.

FLOPS are computed using exact operation counts rather than asymptotic
approximations. The exact formula is stated for each benchmark below.

### Synchronisation

Reactant/XLA kernel dispatch is asynchronous by default. Without explicit
synchronisation, `@elapsed` measures only dispatch latency rather than actual
computation time, producing results that are orders of magnitude too optimistic.

All Julia scripts use `@compile sync=true` to block until TPU execution is
complete before the timer stops:

    compiled = @compile sync=true f(args...)

An earlier approach used `Array(C)[1,1]` as an implicit synchronisation
barrier. This forced a full host-device array transfer on every timed
iteration, introducing overhead that caused a spurious performance regression
at n=3000 and underestimated throughput by up to 57× at large matrix sizes.
Switching to `sync=true` eliminated this artefact entirely.

In Python/JAX, the equivalent is `block_until_ready()`, called on every
timed result.

### Note on Float64

Float64 is excluded from all benchmarks. TPU v4 has no native double-precision
hardware support. JAX silently downcasts Float64 to Float32 on TPU by default,
producing throughput figures approximately double those of Float32 — a
physically impossible result that confirms the silent downcast.
Julia/Reactant preserves the declared type, making a fair comparison
impossible. Float32 and BFloat16 are the only precisions benchmarked.

---

## 01 — Matrix Multiplication

**Scripts:** `matrix_mul.jl`, `matrix_mul.py`, `matrix_mul_cpu.py`

**Formula:** `FLOPS = 2n³ - n²`

**Precisions:** Float32, BFloat16.

**Sizes:** n=100 to n=20,000.

**Finding:** At large matrix sizes (n≥8,000), Julia/Reactant and Python/JAX
perform essentially equivalently for both precisions — within 1% of each
other. At small sizes, JAX is faster due to lower dispatch latency (~0.15 ms
vs ~0.22 ms). BFloat16 throughput is marginally higher than Float32 for both
stacks at large sizes, consistent with TPU v4's native BFloat16 matrix units.
Peak throughput reaches approximately 245–253 TFLOPS for both stacks at
n=20,000.

---

## 02 — Vector Dot Product

**Scripts:** `dot_product.jl`, `dot_product.py`

**Formula:** `FLOPS = 2n - 1`

**Precisions:** Float32, BFloat16.

**Sizes:** n=1,000 to n=10,000,000.

**Finding:** Wall-clock time is essentially flat across all vector sizes for
both stacks, confirming that TPU dispatch latency dominates over arithmetic
throughput at these problem sizes. Julia/Reactant dispatch overhead is
~0.22 ms vs ~0.14 ms for Python/JAX — a consistent 1.5× difference
reflecting the maturity gap between the two runtimes rather than any
difference in TPU hardware throughput.

---

## 03 — Random Number Generation

**Scripts:** `rng_benchmark.jl`, `rng_benchmark.py`

**Formula:** `Throughput (GB/s) = (n × sizeof(T)) / (time_s × 10⁹)`

**Precisions:** Float32, BFloat16.

**Sizes:** n=1,000 to n=10,000,000.

**Finding:** Both stacks are dispatch-dominated at small sizes, transitioning
to bandwidth-limited behaviour around n=500,000–1,000,000. At large sizes
Julia/Reactant outperforms Python/JAX — ~138 GB/s vs ~86 GB/s for Float32
and ~74 GB/s vs ~42 GB/s for BFloat16 at n=10,000,000. JAX throughput
flattens at large sizes, likely due to overhead from mandatory
`jax.random.split` key management on every timed iteration.

---

## 04 — Batched Matrix Multiplication

**Scripts:** `batched_matmul.jl`, `batched_matmul.py`

**Formula:** `FLOPS = b × (2n³ - n²)`

**Precisions:** Float32, BFloat16.

**Matrix size:** Fixed at n=1024.

**Batch sizes:** b=1 to b=512 (powers of 2).

**Finding:** JAX outperforms Julia/Reactant throughout, achieving
approximately 1.6–1.75× higher GFLOPS at peak. Julia shows a performance
drop at b=512 for both precisions, attributable to the comprehension-based
batched implementation rather than a hardware limitation.

---

## 05 — Automatic Differentiation

**Scripts:** `gradient.jl`, `gradient.py`

**Function:** `L(A, B) = sum((A_b * B_b)² for b in 1:batch)`

Gradient dL/dA computed via reverse-mode AD. Julia uses Enzyme.jl integrated
with Reactant; Python uses `jax.grad` with `jit`.

**Formula:** `FLOPS = 3 × b × (2n³ - n²)`

The factor of 3 accounts for one forward pass and approximately two backward
passes (standard convention for reverse-mode AD).

**Precisions:** Float32, BFloat16.

**Matrix size:** Fixed at n=1024.

**Batch sizes:** b=1 to b=512 (powers of 2).

**Finding:** JAX outperforms Julia/Reactant, but Julia is more competitive
here than in the batched matmul benchmark. At peak, Julia achieves ~69% of
JAX throughput for Float32 (~219 vs ~315 TFLOPS) and ~76% for BFloat16
(~279 vs ~368 TFLOPS). Unlike benchmark 04, Julia shows no performance drop
at b=512, suggesting Enzyme's gradient compilation scales more efficiently
than the comprehension-based forward pass.
