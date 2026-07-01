"""
TPU Vector Dot Product Benchmark - Python/JAX
Tests dot product performance on TPU across two precisions:
float32 and bfloat16.

Note: Float64 is excluded because JAX silently downcasts Float64 to Float32
on TPU (unless jax_enable_x64 is explicitly set), producing inflated
throughput figures that cannot be fairly compared against Julia/Reactant,
which preserves the declared type. See BENCHMARK_NOTES.md for details.
"""

import jax
import jax.numpy as jnp
import time
import numpy as np

print("="*70)
print("Python/JAX TPU Vector Dot Product Benchmark")
print("="*70)
print()

print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
print()

if jax.default_backend() != 'tpu':
    print("WARNING: TPU not detected! Running on:", jax.default_backend())
    print("This benchmark is designed for TPU execution.")
else:
    print("✓ TPU detected and ready!")
print()

# Vector sizes to test — extended to show dispatch-dominated regime clearly
sizes = [1_000, 10_000, 100_000, 500_000, 1_000_000, 2_000_000,
         5_000_000, 10_000_000]

# Precisions to test
precisions = {
    "Float32":  jnp.float32,
    "BFloat16": jnp.bfloat16,
}

print("Running TPU benchmarks...")
print()

results = {}

for prec_name, dtype in precisions.items():
    print(f"Precision: {prec_name}")
    print("-"*70)

    for n in sizes:
        print(f"  Benchmarking n={n}... ", end='', flush=True)

        key = jax.random.PRNGKey(0)
        a = jax.random.normal(key, (n,), dtype=dtype)
        b = jax.random.normal(key, (n,), dtype=dtype)

        @jax.jit
        def dot_product(a, b):
            return jnp.dot(a, b)

        # Warm-up
        _ = dot_product(a, b).block_until_ready()

        num_runs = 20 if n < 100_000 else 10 if n < 1_000_000 else 5
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = dot_product(a, b).block_until_ready()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        median_time = np.median(times)
        flops = 2.0 * n - 1.0
        gflops = flops / (median_time * 1e6)

        results[(n, prec_name)] = (median_time, gflops)

        print("✓")
        print(f"  n: {n:10d} | Precision: {prec_name:<10} | "
              f"Time: {median_time:8.3f} ms | GFLOPS: {gflops:10.4f}")

    print()

print()
print("="*70)
print("Summary of TPU Results")
print("="*70)
print()

for prec_name in precisions:
    print(f"Precision: {prec_name}")
    print(f"{'Size':<15} {'Time (ms)':<15} {'GFLOPS':<15}")
    print("-"*45)
    for n in sizes:
        time_ms, gflops = results[(n, prec_name)]
        print(f"{n:<15} {time_ms:>12.3f}    {gflops:>12.4f}")
    print()

with open('python_tpu_results.txt', 'w') as f:
    f.write("size,precision,time_ms,gflops\n")
    for prec_name in precisions:
        for n in sizes:
            time_ms, gflops = results[(n, prec_name)]
            f.write(f"{n},{prec_name},{time_ms},{gflops}\n")

print()
print("Results saved to python_tpu_results.txt")

print()
print("="*70)
print("TPU Device Information")
print("="*70)
for device in jax.devices():
    print(f"Device: {device}")
    print(f"  Platform: {device.platform}")
    print(f"  Device kind: {device.device_kind}")
print()
print("Benchmark completed!")

