"""
TPU Matrix Multiplication Benchmark - Python/JAX
Tests matrix multiplication performance on TPU across two precisions:
float32 and bfloat16.

FLOPS formula: 2n^3 - n^2
"""

import jax
import jax.numpy as jnp
import time
import numpy as np

print("="*70)
print("Python/JAX TPU Matrix Multiplication Benchmark")
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

# Matrix sizes to test
sizes = [100, 500, 1000, 2000, 3000, 5000, 8000, 10000, 20000]

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
        print(f"  Benchmarking {n}×{n}... ", end='', flush=True)

        key = jax.random.PRNGKey(0)
        A = jax.random.normal(key, (n, n), dtype=dtype)
        B = jax.random.normal(key, (n, n), dtype=dtype)

        @jax.jit
        def matmul(a, b):
            return jnp.dot(a, b)

        _ = matmul(A, B).block_until_ready()

        times = []
        num_runs = 20 if n < 2000 else 10 if n < 5000 else 5

        for _ in range(num_runs):
            start = time.perf_counter()
            _ = matmul(A, B).block_until_ready()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        median_time = np.median(times)
        flops = 2.0 * n**3 - n**2
        gflops = flops / (median_time * 1e6)

        results[(n, prec_name)] = (median_time, gflops)

        print("✓")
        print(f"  Size: {n:5d}×{n:<5d} | Time: {median_time:8.3f} ms | GFLOPS: {gflops:10.2f}")

    print()

print()
print("="*70)
print("Summary of TPU Results")
print("="*70)
print()

for prec_name in precisions:
    print(f"Precision: {prec_name}")
    print(f"{'Size':<15} {'Time (ms)':<15} {'GFLOPS':<15} {'TFLOPs/s':<12}")
    print("-"*55)
    for n in sizes:
        time_ms, gflops = results[(n, prec_name)]
        tflops = gflops / 1000.0
        print(f"{n}×{n:<8} {time_ms:>12.3f}    {gflops:>12.2f}    {tflops:>10.3f}")
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
