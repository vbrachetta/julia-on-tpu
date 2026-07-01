<p align="center">
  <img src="assets/julia-on-tpu.svg" width="300" alt="julia-on-tpu logo">
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.XXXXXXX">
    <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue" alt="DOI">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

# Julia on TPU: Reproducible Examples and Benchmarks

This work documents a systematic investigation of the Julia programming language on Google Tensor Processing Unit (TPU) hardware, using Reactant.jl and the XLA compiler as the execution backend. It provides nine self-contained example scripts covering progressively advanced numerical workloads — from basic array operations and operator fusion to in-place mutation, data reductions, stencil computations, and reverse-mode automatic differentiation via Enzyme.jl — together with five quantitative benchmarks comparing Julia/Reactant against Python/JAX across matrix multiplication, vector dot product, random number generation, batched matrix multiplication, and gradient computation workloads. A containerised execution environment is provided for Docker, Podman, and Apptainer, alongside full documentation of installation methods and known issues on Google Cloud TPU VMs. A key methodological finding is that Reactant/XLA kernel dispatch is asynchronous by default, and naive timing without explicit synchronisation underestimates wall-clock time by up to 57× at large problem sizes. To the author's knowledge, this is one of the first publicly available and reproducible demonstrations of Julia running natively on TPU hardware.

**Author:** Vincenzo Brachetta, University of Birmingham (UK)  
**Contact:** v.brachetta@bham.ac.uk

---

## Table of Contents

- [Background](#background)
- [Requirements](#requirements)
- [Examples](#examples)
- [Benchmarks](#benchmarks)
- [Outputs](#outputs)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Direct](#direct-julia-installed-locally)
  - [Docker](#docker)
  - [Podman](#podman)
  - [Apptainer](#apptainer)
- [Notes on Precision](#notes-on-precision)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [Licence](#licence)

---

## Background

[Julia](https://julialang.org/) is a high-level, high-performance programming language for numerical and scientific computing, combining ease of use with execution speeds approaching C and Fortran via just-in-time compilation through LLVM.

[Reactant.jl](https://enzymead.github.io/Reactant.jl/stable/) is a Julia compiler framework that traces Julia functions and lowers them to [MLIR](https://mlir.llvm.org/), from which [XLA](https://openxla.org/xla) generates optimised executables for CPUs, GPUs, and TPUs. Ordinary Julia code can be compiled into hardware-specific kernels with minimal modification via the `@compile` macro.

[Enzyme.jl](https://enzyme.mit.edu/) is a Julia package for automatic differentiation that operates at the compiler level. When used with Reactant.jl, it compiles reverse-mode automatic differentiation directly into the XLA computation graph, enabling gradient computation at full hardware throughput on the TPU.

---

## Requirements

- Julia >= 1.12
- Docker >= 29.3.0, Podman >= 3.4.4, or Apptainer >= 1.0 (see Usage below)
- Access to a TPU device (e.g. via Google TPU Research Cloud)

When using Docker, Podman, or Apptainer, Julia package installation is handled inside the container — no manual `Pkg.add` is required. For a local Julia installation, the required packages are:

```julia
using Pkg
Pkg.add(["Reactant", "BFloat16s", "Enzyme", "BenchmarkTools"])
```

> **Note:** `Enzyme.jl` is only required for `06_automatic_differentiation.jl` and the gradient benchmark. `BFloat16s.jl` is only required for `03_matrix_mul_bfloat16.jl`.

### Tested versions

| Component | Version |
|-----------|---------|
| Julia | 1.12.5 |
| Reactant.jl | 0.2.234 |
| Enzyme.jl | 0.13.131 |
| BenchmarkTools.jl | 1.6.3 |
| Docker | 29.2.1 |
| Podman | 3.4.4 |
| Apptainer | 1.4.1 |
| Python | 3.10.12 |
| JAX | 0.6.2 |
| NumPy | 1.26.2 |

The Python/JAX versions are relevant only for the benchmark comparison scripts in `benchmarks/`, which read results from pre-generated text files and do not require a JAX installation to run.

---

## Examples

Nine self-contained Julia scripts are provided in `examples/`, covering progressively advanced TPU workloads. The captured outputs in `examples/out/` provide a reference for expected results.

| File | Topic |
|------|-------|
| `01_reactant_test.jl` | Backend initialisation and basic compiled addition |
| `02_matrix_mul_float32.jl` | Matrix multiplication (Float32) |
| `03_matrix_mul_bfloat16.jl` | Matrix multiplication (BFloat16) |
| `04_operator_fusion.jl` | Fused element-wise kernels |
| `05_inplace_mutation.jl` | In-place buffer updates |
| `06_automatic_differentiation.jl` | Reverse-mode AD via Enzyme.jl |
| `07_data_reductions.jl` | Global reductions (sum, mean, max) |
| `08_stencil_computation.jl` | 2D stencil (Laplacian) |
| `09_tpu_random_gen.jl` | Device-side random number generation |

---

## Benchmarks

Five benchmarks compare Julia/Reactant against Python/JAX on a Google Cloud
TPU v4 VM. Each benchmark folder contains the Julia script, the Python/JAX
equivalent, pre-generated result files, a plotting script, and output figures.

| Folder | Workload | Metric | Precisions |
|--------|----------|--------|------------|
| `01_matrix_mul` | Matrix multiplication | GFLOPS | Float32, BFloat16 |
| `02_dot_product` | Vector dot product | GFLOPS, time (ms) | Float32, BFloat16 |
| `03_random` | Random number generation | GB/s, time (ms) | Float32, BFloat16 |
| `04_batched_matmul` | Batched matrix multiplication | GFLOPS | Float32, BFloat16 |
| `05_gradient` | Reverse-mode AD (Enzyme vs JAX) | GFLOPS | Float32, BFloat16 |

Full methodology and per-benchmark notes are provided in
[`benchmarks/benchmark_notes.md`](benchmarks/benchmark_notes.md).

---

## Outputs

The `examples/out/` directory contains the terminal output of each example script, captured on a Google Cloud TPU v4 VM, and numbered to correspond to the examples in `examples/`. These files serve as a reference for expected results and can be used to verify correctness without access to TPU hardware.

---

## Project Structure

```
.
├── examples/                  # Nine self-contained Julia example scripts
│   └── out/                   # Captured terminal outputs for each example
├── benchmarks/                # Benchmark scripts, results, and plots
│   ├── 01_matrix_mul/         # Matrix multiplication benchmark
│   ├── 02_dot_product/        # Vector dot product benchmark
│   ├── 03_random/             # Random number generation benchmark
│   ├── 04_batched_matmul/     # Batched matrix multiplication benchmark
│   ├── 05_gradient/           # Automatic differentiation benchmark
│   └── benchmark_notes.md     # Methodology and per-benchmark observations
├── assets/                    # Logo and source artwork
├── installation/              # Installation documentation and container files
│   ├── dockerfile             # Docker/Podman container definition
│   ├── julia.def              # Apptainer container definition
│   ├── install_apptainer.sh   # Apptainer installer for Ubuntu (PPA method)
│   ├── installation.md        # Full installation reference for all methods
│   ├── test_julia.jl          # Package load verification script
│   ├── dumps/                 # Terminal dumps from failed installation attempts
│   ├── run_julia_docker.sh    # Run a script in the Docker container
│   ├── julia_repl_docker.sh   # Start interactive REPL in the Docker container
│   ├── run_julia_podman.sh    # Run a script in the Podman container
│   ├── julia_repl_podman.sh   # Start interactive REPL in the Podman container
│   ├── run_julia_apptainer.sh # Run a script in the Apptainer container
│   └── julia_repl_apptainer.sh # Start interactive REPL in the Apptainer container
├── CITATION.cff               # Citation metadata
├── LICENSE                    # MIT Licence
└── README.md                  # This file
```

---

## Usage

A complete account of all installation methods tested — including native installation via juliaup and snap, Docker, Podman, and Apptainer — together with known issues, workarounds, and troubleshooting notes, is provided in [`installation/installation.md`](installation/installation.md).

### Direct (Julia installed locally)

```bash
julia examples/01_reactant_test.jl
```

All scripts call `Reactant.set_default_backend("tpu")` at the top. On a machine without a TPU this will fail — use a TRC VM or modify the backend to `"cpu"` for local testing. Native installation on the TPU VM has known issues: both juliaup and snap suffer from a `double free or corruption` crash in `Pkg` originating in Julia's bundled `libcurl`, and no working workaround was found. Containerised methods are recommended.

### Docker

```bash
sudo docker build -t julia-tpu-image installation/
./installation/run_julia_docker.sh examples/01_reactant_test.jl
./installation/julia_repl_docker.sh
```

### Podman

```bash
podman build -t julia-tpu-image -f installation/dockerfile \
    --from docker.io/library/julia:1.12.5 installation/
./installation/run_julia_podman.sh examples/01_reactant_test.jl
./installation/julia_repl_podman.sh
```

### Apptainer

If Apptainer is not yet installed, a convenience script is provided:

```bash
sudo installation/install_apptainer.sh
```

```bash
apptainer build installation/julia.sif installation/julia.def
./installation/run_julia_apptainer.sh examples/01_reactant_test.jl
./installation/julia_repl_apptainer.sh
```

---

## Notes on Precision

TPUs natively favour **BFloat16** (1 sign bit, 8 exponent bits, 7 mantissa bits) for matrix operations. Its 8-exponent-bit layout matches Float32, preserving the same dynamic range, but the reduced mantissa yields approximately 2–3 significant decimal digits compared to Float32's 7. This trade-off is well suited to machine learning workloads but should be considered carefully in scientific computing contexts where numerical accuracy is critical. Both Float32 and BFloat16 workflows are demonstrated in the example scripts and benchmarks.

Float64 is excluded from all benchmarks. TPU v4 has no native double-precision hardware support, and JAX silently downcasts Float64 to Float32 on TPU by default, making a fair comparison impossible. Full details are in [`benchmarks/benchmark_notes.md`](benchmarks/benchmark_notes.md).

---

## Acknowledgements

This work was carried out as part of the project *Accelerating Data-Intensive
Research with Google TPU Infrastructure* at the University of Birmingham (UK).

TPU resources were provided through the
[Google TPU Research Cloud (TRC)](https://sites.research.google/trc/about/)
programme. The author gratefully acknowledges Google's support.

This work was developed with the assistance of Claude Sonnet 4.6 (Anthropic)
for code generation and documentation drafting. All code and content were
reviewed, validated, and edited by the author.

The author wishes to thank **Prof. Mayorkinos Papaelias** for providing the
time and support that made this work possible.

---

## Citation

If you use this work in your research, please cite it as follows:

```bibtex
@software{brachetta2026julia_on_tpu,
  author    = {Brachetta, Vincenzo},
  title     = {Julia on TPU: Reproducible Examples and Benchmarks},
  year      = {2026},
  publisher = {Zenodo},
  url       = {https://github.com/vbrachetta/julia-on-tpu},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

A [`CITATION.cff`](CITATION.cff) file is also provided.

---

## Licence

This project is licensed under the MIT Licence. See the [LICENSE](LICENSE)
file for details.
