import numpy as np
import time
import csv

sizes = [100, 500, 1000, 2000, 3000, 5000, 8000, 10000, 20000]

print("=" * 70)
print("NumPy CPU Matrix Multiplication Benchmark")
print("=" * 70)
print(f"NumPy version: {np.__version__}")

# BLAS info API changed in NumPy 2.0
try:
    blas_info = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
except Exception:
    try:
        blas_info = np.__config__.blas_opt_info.get("libraries", "unknown")
    except Exception:
        blas_info = "unavailable"
print(f"NumPy BLAS:    {blas_info}")
print()

results = []

for n in sizes:
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)

    # Warm-up
    _ = A @ B

    num_runs = 20 if n < 2000 else (10 if n < 5000 else 5)
    times = []

    for _ in range(num_runs):
        t0 = time.perf_counter()
        C = A @ B
        # NumPy matmul is synchronous — no barrier needed
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    med    = float(np.median(times))
    flops  = 2.0 * n**3 - n**2
    gflops = flops / (med * 1e6)
    results.append((n, med, gflops))

    print(f"  {n:5d}x{n:<5d} | {med:8.3f} ms | {gflops:12.2f} GFLOPS")

print()
with open("python_cpu_results.txt", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["size", "time_ms", "gflops"])
    writer.writerows(results)

print("Results saved to python_cpu_results.txt")
print("=" * 70)
