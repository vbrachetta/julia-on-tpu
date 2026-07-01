#=
rng_benchmark.jl - TPU Random Number Generation Benchmark
Tests on-device random number generation performance on TPU using
Reactant/XLA across two precisions: Float32 and BFloat16.
Synchronisation is handled via the sync=true option in @compile, which blocks
until the TPU has completed execution without incurring a full host-device
array transfer on every measurement.
Results are saved to julia_tpu_results.txt.

Throughput is reported in GB/s: the number of bytes written (n * sizeof(T))
divided by the median wall-clock time. This is the natural metric for RNG,
which is a memory-bound operation with no meaningful arithmetic intensity.

The RNG is compiled into the XLA graph by tracing through Julia's rand()
inside a @compile block, using the shape of a template array to define the
output size — the same pattern used in 09_tpu_random_gen.jl.

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
println("Julia/Reactant TPU Random Number Generation Benchmark")
println("="^70)
println("Julia version: ", VERSION)
println()

# Vector sizes to test — extended to large sizes where bandwidth dominates
# over XLA dispatch latency (~0.22 ms floor on TPU v4)
sizes = [1_000, 10_000, 100_000, 500_000, 1_000_000, 2_000_000,
         5_000_000, 10_000_000]

# Precisions to test
precisions = [Float32, BFloat16]

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

        template = Reactant.to_rarray(zeros(T, n))

        generate_random(x) = rand(T, size(x)...)

        compiled = @compile sync=true generate_random(template)

        _ = compiled(template)

        println("✓  Benchmarking...")
        flush(stdout)

        num_runs = n < 100_000 ? 20 : (n < 1_000_000 ? 10 : 5)
        times = Float64[]
        for _ in 1:num_runs
            t = @elapsed compiled(template)
            push!(times, t * 1000)
        end

        med   = median(times)
        bytes = n * sizeof(T)
        gb_s  = (bytes / 1e9) / (med / 1e3)

        results[(n, T)] = (med, gb_s)

        @printf("  n: %10d | Precision: %-10s | Median: %8.3f ms | GB/s: %10.4f\n",
                n, T, med, gb_s)
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
    @printf("%-15s %-15s %-15s\n", "Size", "Time (ms)", "GB/s")
    println("-"^45)
    for n in sizes
        med, gb_s = results[(n, T)]
        @printf("%-15d %12.3f    %12.4f\n", n, med, gb_s)
    end
    println()
end

open("julia_tpu_results.txt", "w") do f
    println(f, "size,precision,time_ms,gb_s")
    for T in precisions
        for n in sizes
            med, gb_s = results[(n, T)]
            println(f, "$n,$T,$med,$gb_s")
        end
    end
end

println()
println("Results saved to julia_tpu_results.txt")
println("="^70)

