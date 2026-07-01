#=
batched_matmul.jl - TPU Batched Matrix Multiplication Benchmark
Tests batched matrix multiplication performance on TPU using Reactant/XLA
across two precisions: Float32 and BFloat16.
Matrix size is fixed at n=1024; batch size varies from 1 to 512.
Synchronisation is handled via the sync=true option in @compile, which blocks
until the TPU has completed execution without incurring a full host-device
array transfer on every measurement.
Results are saved to julia_tpu_results.txt.

FLOPS formula: b × (2n³ - n²)
For a batch of b matrix multiplications of size n×n, each output element
requires n multiplications and n-1 additions, giving 2n³-n² operations
per matrix and b×(2n³-n²) in total.
=#

using Reactant
using BFloat16s
using Statistics
using Printf

Reactant.set_default_backend("tpu")

println("="^70)
println("Julia/Reactant TPU Batched Matrix Multiplication Benchmark")
println("="^70)
println("Julia version: ", VERSION)
println()

const N = 1024

# Batch sizes to test
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Precisions to test
precisions = [Float32, BFloat16]

# Batched matmul: A is (n, n, b), B is (n, n, b)
# We use broadcasting over the batch dimension
function batched_matmul(A, B)
    return stack([A[:, :, i] * B[:, :, i] for i in axes(A, 3)])
end

println("="^70)
println("Julia TPU Benchmarks (n=$(N)×$(N))")
println("="^70)
println()

results = Dict{Tuple{Int, DataType}, Tuple{Float64, Float64}}()

for T in precisions
    println("Precision: $T")
    println("-"^70)

    for b in batch_sizes
        print("Compiling batch=$b... ")
        flush(stdout)

        A_host = rand(T, N, N, b)
        B_host = rand(T, N, N, b)

        A_tpu = Reactant.to_rarray(A_host)
        B_tpu = Reactant.to_rarray(B_host)

        compiled = @compile sync=true batched_matmul(A_tpu, B_tpu)

        _ = compiled(A_tpu, B_tpu)

        println("✓  Benchmarking...")
        flush(stdout)

        num_runs = b < 16 ? 20 : (b < 128 ? 10 : 5)
        times = Float64[]
        for _ in 1:num_runs
            t = @elapsed compiled(A_tpu, B_tpu)
            push!(times, t * 1000)
        end

        med    = median(times)
        flops  = b * (2.0 * N^3 - N^2)
        gflops = flops / (med * 1e6)

        results[(b, T)] = (med, gflops)

        @printf("  batch: %4d | Precision: %-10s | Median: %8.3f ms | GFLOPS: %10.2f\n",
                b, T, med, gflops)
    end
    println()
end

println()
println("="^70)
println("Summary")
println("="^70)
println()

for T in precisions
    println("Precision: $T  (n=$(N)×$(N))")
    @printf("%-12s %-15s %-15s\n", "Batch", "Time (ms)", "GFLOPS")
    println("-"^42)
    for b in batch_sizes
        med, gflops = results[(b, T)]
        @printf("%-12d %12.3f    %12.2f\n", b, med, gflops)
    end
    println()
end

open("julia_tpu_results.txt", "w") do f
    println(f, "batch,precision,time_ms,gflops")
    for T in precisions
        for b in batch_sizes
            med, gflops = results[(b, T)]
            println(f, "$b,$T,$med,$gflops")
        end
    end
end

println()
println("Results saved to julia_tpu_results.txt")
println("="^70)

