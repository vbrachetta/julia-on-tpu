"""
TPU Batched Matrix Multiplication Benchmark - Python/JAX
Tests batched matrix multiplication performance on TPU across two
precisions: float32 and bfloat16.
Matrix size is fixed at n=1024; batch size varies from 1 to 512.

FLOPS formula: b x (2n^3 - n^2)
For a batch of b matrix multiplications of size n×n, each output element
requires n multiplications and n-1 additions, giving 2n^3-n^2 operations
per matrix and b×(2n^3-n^2) in total.
"""

import jax
import jax.numpy as jnp
import time
import numpy as np

print("="*70)
print("Python/JAX TPU Batched Matrix Multiplication Benchmark")
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

N = 1024

# Batch sizes to test
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Precisions to test
precisions = {
    "Float32":  jnp.float32,
    "BFloat16": jnp.bfloat16,
}

print(f"Running TPU benchmarks (n={N}×{N})...")
print()

results = {}

for prec_name, dtype in precisions.items():
    print(f"Precision: {prec_name}")
    print("-"*70)

    key = jax.random.PRNGKey(0)

    for b in batch_sizes:
        print(f"  Benchmarking batch={b}... ", end='', flush=True)

        A = jax.random.normal(key, (b, N, N), dtype=dtype)
        B = jax.random.normal(key, (b, N, N), dtype=dtype)

        @jax.jit
        def batched_matmul(A, B):
            return jnp.matmul(A, B)

        _ = batched_matmul(A, B).block_until_ready()

        num_runs = 20 if b < 16 else 10 if b < 128 else 5
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = batched_matmul(A, B).block_until_ready()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        median_time = np.median(times)
        flops = b * (2.0 * N**3 - N**2)
        gflops = flops / (median_time * 1e6)

        results[(b, prec_name)] = (median_time, gflops)

        print("✓")
        print(f"  batch: {b:4d} | Precision: {prec_name:<10} | "
              f"Time: {median_time:8.3f} ms | GFLOPS: {gflops:10.2f}")

    print()

print()
print("="*70)
print("Summary of TPU Results")
print("="*70)
print()

for prec_name in precisions:
    print(f"Precision: {prec_name}  (n={N}×{N})")
    print(f"{'Batch':<12} {'Time (ms)':<15} {'GFLOPS':<15}")
    print("-"*42)
    for b in batch_sizes:
        time_ms, gflops = results[(b, prec_name)]
        print(f"{b:<12} {time_ms:>12.3f}    {gflops:>12.2f}")
    print()

with open('python_tpu_results.txt', 'w') as f:
    f.write("batch,precision,time_ms,gflops\n")
    for prec_name in precisions:
        for b in batch_sizes:
            time_ms, gflops = results[(b, prec_name)]
            f.write(f"{b},{prec_name},{time_ms},{gflops}\n")

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

