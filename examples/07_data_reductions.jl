#=
Example 07 - Global Data Reductions
Demonstrates sum, mean, and maximum reductions across a large
matrix using XLA's tree-based reduction strategy.
=#

using Reactant
using Statistics

Reactant.set_default_backend("tpu")

function reduction_ops(x)
    total_sum = sum(x)
    avg_val   = mean(x)
    max_val   = maximum(x)
    return total_sum, avg_val, max_val
end

data_tpu = Reactant.to_rarray(rand(Float32, 2048, 2048))

reduce_compiled = @compile reduction_ops(data_tpu)
s, m, mx = reduce_compiled(data_tpu)

println("Example 07: Reductions complete.")
println("Sum: $s, Mean: $m, Max: $mx")

