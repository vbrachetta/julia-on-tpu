"""
TPU Random Number Generation Benchmark - Python/JAX
Tests on-device random number generation performance on TPU across two
precisions: float32 and bfloat16.

Throughput is reported in GB/s: the number of bytes written (n * itemsize)
divided by the median wall-clock time. This is the natural metric for RNG,
which is a memory-bound operation with no meaningful arithmetic intensity.

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
print("Python/JAX TPU Random Number Generation Benchmark")
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

# Vector sizes to test — extended to large sizes where bandwidth dominates
# over JAX dispatch latency (~0.14 ms floor on TPU v4)
sizes = [1_000, 10_000, 100_000, 500_000, 1_000_000, 2_000_000,
         5_000_000, 10_000_000]

# Precisions to test
precisions = {
    "Float32":  (jnp.float32,  4),
    "BFloat16": (jnp.bfloat16, 2),
}

print("Running TPU benchmarks...")
print()

results = {}

for prec_name, (dtype, itemsize) in precisions.items():
    print(f"Precision: {prec_name}")
    print("-"*70)

    key = jax.random.PRNGKey(0)

    for n in sizes:
        print(f"  Benchmarking n={n}... ", end='', flush=True)

        @jax.jit
        def generate_random(key, n=n, dtype=dtype):
            return jax.random.uniform(key, shape=(n,), dtype=dtype)

        _ = generate_random(key).block_until_ready()

        num_runs = 20 if n < 100_000 else 10 if n < 1_000_000 else 5
        times = []
        for _ in range(num_runs):
            key, subkey = jax.random.split(key)
            start = time.perf_counter()
            _ = generate_random(subkey).block_until_ready()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        median_time = np.median(times)
        bytes_written = n * itemsize
        gb_s = (bytes_written / 1e9) / (median_time / 1e3)

        results[(n, prec_name)] = (median_time, gb_s)

        print("✓")
        print(f"  n: {n:10d} | Precision: {prec_name:<10} | "
              f"Time: {median_time:8.3f} ms | GB/s: {gb_s:10.4f}")

    print()

print()
print("="*70)
print("Summary of TPU Results")
print("="*70)
print()

for prec_name in precisions:
    print(f"Precision: {prec_name}")
    print(f"{'Size':<15} {'Time (ms)':<15} {'GB/s':<15}")
    print("-"*45)
    for n in sizes:
        time_ms, gb_s = results[(n, prec_name)]
        print(f"{n:<15} {time_ms:>12.3f}    {gb_s:>12.4f}")
    print()

with open('python_tpu_results.txt', 'w') as f:
    f.write("size,precision,time_ms,gb_s\n")
    for prec_name in precisions:
        for n in sizes:
            time_ms, gb_s = results[(n, prec_name)]
            f.write(f"{n},{prec_name},{time_ms},{gb_s}\n")

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

