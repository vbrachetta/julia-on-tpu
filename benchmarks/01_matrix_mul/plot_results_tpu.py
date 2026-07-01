"""
plot_results_tpu.py - Plot matrix multiplication benchmark comparison
(Julia/Reactant vs Python/JAX on TPU v4).
Reads julia_tpu_results.txt and python_tpu_results.txt and produces
comparison plots for wall-clock time and GFLOPS, one panel per precision.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

julia_df  = pd.read_csv("julia_tpu_results.txt")
python_df = pd.read_csv("python_tpu_results.txt")

precisions = ["Float32", "BFloat16"]

julia_df["precision"] = julia_df["precision"].str.capitalize()
julia_df["precision"] = julia_df["precision"].replace({"Bfloat16": "BFloat16"})

# --- Plot: GFLOPS comparison ----------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.suptitle("Matrix Multiplication: Julia/Reactant vs Python/JAX on TPU v4 — GFLOPS",
             fontsize=13, fontweight="bold")

for ax, prec in zip(axes, precisions):
    j = julia_df[julia_df["precision"] == prec]
    p = python_df[python_df["precision"] == prec]

    ax.plot(j["size"], j["gflops"] / 1e3, marker="s", label="Julia/Reactant",
            color="royalblue", linewidth=2)
    ax.plot(p["size"], p["gflops"] / 1e3, marker="o", label="Python/JAX",
            color="darkorange", linewidth=2)

    ax.set_title(f"Precision: {prec}", fontsize=12)
    ax.set_xlabel("Matrix size (n×n)", fontsize=10)
    ax.set_ylabel("TFLOPS", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="x", rotation=35)

plt.tight_layout()
plt.savefig("benchmark_comparison_gflops.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved benchmark_comparison_gflops.png")

# --- Plot: wall-clock time comparison -------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.suptitle("Matrix Multiplication: Julia/Reactant vs Python/JAX on TPU v4 — Wall-Clock Time",
             fontsize=13, fontweight="bold")

for ax, prec in zip(axes, precisions):
    j = julia_df[julia_df["precision"] == prec]
    p = python_df[python_df["precision"] == prec]

    ax.plot(j["size"], j["time_ms"], marker="s", label="Julia/Reactant",
            color="royalblue", linewidth=2)
    ax.plot(p["size"], p["time_ms"], marker="o", label="Python/JAX",
            color="darkorange", linewidth=2)

    ax.set_title(f"Precision: {prec}", fontsize=12)
    ax.set_xlabel("Matrix size (n×n)", fontsize=10)
    ax.set_ylabel("Median time (ms)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="x", rotation=35)

plt.tight_layout()
plt.savefig("benchmark_comparison_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved benchmark_comparison_time.png")

print("Plotting complete.")
