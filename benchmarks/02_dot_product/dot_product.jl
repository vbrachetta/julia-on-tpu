#=
dot_product.jl - TPU Vector Dot Product Benchmark
Tests dot product performance on TPU using Reactant/XLA across two
precisions: Float32 and BFloat16.
Synchronisation is handled via the sync=true option in @compile, which blocks
until the TPU has completed execution without incurring a full host-device
array transfer on every measurement.
Results are saved to julia_tpu_results.txt.

Note: Float64 is excluded because TPU v4 has no native double-precision
hardware support. JAX silently downcasts Float64 to Float32 on TPU, producing
inflated throughput figures. Reactant preserves the declared type, making a
fair comparison impossible. See BENCHMARK_NOTES.md for details.
=#

using Reactant
using BFloat16s
using Statistics
using Printf

Reactant.set_default_backend("tpu")

println("="^70)
println("Julia/Reactant TPU Vector Dot Product Benchmark")
println("="^70)
println("Julia version: ", VERSION)
println()

# Vector sizes to test — extended to show dispatch-dominated regime clearly
sizes = [1_000, 10_000, 100_000, 500_000, 1_000_000, 2_000_000,
         5_000_000, 10_000_000]

# Precisions to test
precisions = [Float32, BFloat16]

# Define dot product once, outside the loop
function dot_product(a, b)
    return sum(a.* b)
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
        print("Compiling n=$n... ")
        flush(stdout)

        a_host = rand(T, n)
        b_host = rand(T, n)

        # Move to TPU before compilation so shape and dtype are fixed
        a_tpu = Reactant.to_rarray(a_host)
        b_tpu = Reactant.to_rarray(b_host)

        # sync=true blocks until TPU execution is complete — correct for benchmarking
        compiled = @compile sync=true dot_product(a_tpu, b_tpu)

        # Warm-up
        _ = compiled(a_tpu, b_tpu)

        println("✓  Benchmarking...")
        flush(stdout)

        num_runs = n < 100_000 ? 20 : (n < 1_000_000 ? 10 : 5)
        times = Float64[]
        for _ in 1:num_runs
            t = @elapsed compiled(a_tpu, b_tpu)
            push!(times, t * 1000)
        end

        med    = median(times)
        flops  = 2.0 * n - 1.0
        gflops = flops / (med * 1e6)

        results[(n, T)] = (med, gflops)

        @printf("  n: %10d | Precision: %-10s | Median: %8.3f ms | GFLOPS: %10.4f\n",
                n, T, med, gflops)
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
    @printf("%-15s %-15s %-15s\n", "Size", "Time (ms)", "GFLOPS")
    println("-"^45)
    for n in sizes
        med, gflops = results[(n, T)]
        @printf("%-15d %12.3f    %12.4f\n", n, med, gflops)
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

