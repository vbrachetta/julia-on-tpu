#=
Example 04 - Operator Fusion
Demonstrates how Reactant fuses multiple element-wise operations
into a single XLA kernel, reducing HBM latency.
=#

using Reactant

Reactant.set_default_backend("tpu")

function fused_math_kernel(x, y)
    return sin.(x) .* exp.(-y) .+ 1.0f0
end

x_tpu = Reactant.to_rarray(rand(Float32, 1024, 1024))
y_tpu = Reactant.to_rarray(rand(Float32, 1024, 1024))

fused_compiled = @compile fused_math_kernel(x_tpu, y_tpu)
result = fused_compiled(x_tpu, y_tpu)

println("Example 04: Operator fusion complete.")

