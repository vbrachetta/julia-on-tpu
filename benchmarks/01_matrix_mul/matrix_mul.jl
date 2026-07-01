#=
matrix_mul.jl - TPU Matrix Multiplication Benchmark
Tests matrix multiplication performance on TPU using Reactant/XLA across
two precisions: Float32 and BFloat16.
Synchronisation is handled via the sync=true option in @compile, which blocks
until the TPU has completed execution without incurring a full host-device
array transfer on every measurement.
Results are saved to julia_tpu_results.txt.

FLOPS formula: 2n³ - n²
=#

using Reactant
using BFloat16s
using Statistics
using Printf

Reactant.set_default_backend("tpu")

println("="^70)
println("Julia/Reactant TPU Matrix Multiplication Benchmark")
println("="^70)
println("Julia version: ", VERSION)
println()

# Matrix sizes to test
sizes = [100, 500, 1000, 2000, 3000, 5000, 8000, 10000, 20000]

# Precisions to test
precisions = [Float32, BFloat16]

# Define matmul once, outside the loop
function matmul(A, B)
    return A * B
end

println("="^70)
println("Julia TPU Benchmarks")
println("="^70)
println()

results = Dict{Tuple{Int, DataType}, Tuple{Float64, Float64}}()

for T in precisions
    println("Precision: $T")
    println("-"^70)

    for n in sizes
        print("Compiling $(n)×$(n)... ")
        flush(stdout)

        A_host = rand(T, n, n)
        B_host = rand(T, n, n)

        A_tpu = Reactant.to_rarray(A_host)
        B_tpu = Reactant.to_rarray(B_host)

        compiled = @compile sync=true matmul(A_tpu, B_tpu)

        _ = compiled(A_tpu, B_tpu)

        println("✓  Benchmarking...")
        flush(stdout)

        num_runs = n < 2000 ? 20 : (n < 5000 ? 10 : 5)
        times = Float64[]

        for _ in 1:num_runs
            t = @elapsed compiled(A_tpu, B_tpu)
            push!(times, t * 1000)
        end

        med    = median(times)
        flops  = 2.0 * n^3 - n^2
        gflops = flops / (med * 1e6)
        results[(n, T)] = (med, gflops)

        @printf("  Size: %5d×%-5d | Median: %8.3f ms | GFLOPS: %10.2f\n",
                n, n, med, gflops)
    end
    println()
end

println()
println("="^70)
println("Summary")
println("="^70)
println()

for T in precisions
    println("Precision: $T")
    @printf("%-15s %-15s %-15s %-12s\n", "Size", "Time (ms)", "GFLOPS", "TFLOPs/s")
    println("-"^55)
    for n in sizes
        med, gflops = results[(n, T)]
        @printf("%d×%-8d %12.3f    %12.2f    %10.3f\n",
                n, n, med, gflops, gflops / 1000.0)
    end
    println()
end

open("julia_tpu_results.txt", "w") do f
    println(f, "size,precision,time_ms,gflops")
    for T in precisions
        for n in sizes
            med, gflops = results[(n, T)]
            println(f, "$n,$T,$med,$gflops")
        end
    end
end

println()
println("Results saved to julia_tpu_results.txt")
println("="^70)
