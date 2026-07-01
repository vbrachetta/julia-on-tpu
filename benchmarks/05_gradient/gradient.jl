#=
gradient.jl - TPU Automatic Differentiation Benchmark
Tests reverse-mode automatic differentiation performance on TPU using
Reactant/XLA and Enzyme.jl across two precisions: Float32 and BFloat16.
Matrix size is fixed at n=1024; batch size varies from 1 to 512.
Synchronisation is handled via the sync=true option in @compile, which blocks
until the TPU has completed execution without incurring a full host-device
array transfer on every measurement.
Results are saved to julia_tpu_results.txt.

The function differentiated is a batched matrix multiplication followed by
a sum-of-squares loss:

    L(A, B) = sum((A_b * B_b)^2 for b in 1:batch)

The gradient dL/dA is computed via Enzyme.jl integrated with Reactant,
which compiles the reverse-mode AD directly into the XLA graph.

FLOPS formula: 3 × b × (2n³ - n²)
The factor of 3 accounts for 1 forward pass and approximately 2 backward
passes, following the standard convention for reverse-mode AD cost.
=#

using Reactant
using Enzyme
using BFloat16s
using Statistics
using Printf

Reactant.set_default_backend("tpu")

println("="^70)
println("Julia/Reactant TPU Automatic Differentiation Benchmark")
println("="^70)
println("Julia version: ", VERSION)
println()

const N = 1024

# Batch sizes to test
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Precisions to test
precisions = [Float32, BFloat16]

# Batched matrix multiplication loss: sum of squares of batched matmul result
function loss(A, B)
    result = stack([A[:, :, i] * B[:, :, i] for i in axes(A, 3)])
    return sum(result .^ 2)
end

# Gradient of loss with respect to A
function compute_grad(A, B)
    return Enzyme.gradient(Reverse, loss, A, B)[1]
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

        compiled = @compile sync=true compute_grad(A_tpu, B_tpu)

        # Warm-up
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
        flops  = 3.0 * b * (2.0 * N^3 - N^2)
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
