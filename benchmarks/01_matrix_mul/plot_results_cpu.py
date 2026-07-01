import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# --- Load results ---
# Each file is optional: the plot includes whichever files are present
def load(path):
    return pd.read_csv(path) if os.path.exists(path) else None

python_df = load("python_tpu_results.txt")
julia_df  = load("julia_tpu_results.txt")
cpu_df    = load("python_cpu_results.txt")

if python_df is None and julia_df is None and cpu_df is None:
    raise FileNotFoundError("No result files found. Run the benchmarks first.")

# --- Find common sizes across all available datasets ---
all_dfs = [df for df in [python_df, julia_df, cpu_df] if df is not None]
shared_sizes = sorted(set.intersection(*[set(df["size"]) for df in all_dfs]))

def filter_df(df):
    if df is None:
        return None
    return df[df["size"].isin(shared_sizes)].sort_values("size")

py  = filter_df(python_df)
ju  = filter_df(julia_df)
cpu = filter_df(cpu_df)

labels = [f"{n:,}" for n in shared_sizes]

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Matrix multiplication benchmark: TPU vs CPU",
             fontsize=13, fontweight="bold", y=1.02)

def plot_series(ax, df, col, label, color, marker):
    if df is not None:
        ax.plot(labels, df[col], marker=marker, linewidth=2,
                label=label, color=color)

# --- Throughput ---
plot_series(ax1, py,  "gflops", "Python/JAX (TPU)",     "#1D9E75", "o")
plot_series(ax1, ju,  "gflops", "Julia/Reactant (TPU)", "#378ADD", "s")
plot_series(ax1, cpu, "gflops", "Python/NumPy (CPU)",           "#D85A30", "^")
ax1.set_title("Throughput (GFLOPS)", fontsize=11)
ax1.set_xlabel("Matrix size (n x n)")
ax1.set_ylabel("GFLOPS")
ax1.legend()
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.tick_params(axis="x", rotation=35)

# --- Time ---
plot_series(ax2, py,  "time_ms", "Python/JAX (TPU)",     "#1D9E75", "o")
plot_series(ax2, ju,  "time_ms", "Julia/Reactant (TPU)", "#378ADD", "s")
plot_series(ax2, cpu, "time_ms", "NumPy (CPU)",           "#D85A30", "^")
ax2.set_title("Median execution time (ms)", fontsize=11)
ax2.set_xlabel("Matrix size (n x n)")
ax2.set_ylabel("Time (ms)")
ax2.legend()
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax2.tick_params(axis="x", rotation=35)

fig.tight_layout()
fig.savefig("benchmark_comparison_cpu.png", dpi=150, bbox_inches="tight")
print("Saved: benchmark_comparison_cpu.png")
