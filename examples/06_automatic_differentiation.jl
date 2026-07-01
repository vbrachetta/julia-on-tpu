#=
Example 06 - Reverse-Mode Automatic Differentiation
Integrates Enzyme.jl with Reactant to compile gradient computation
directly into the XLA graph for full-throughput TPU execution.
=#

using Reactant, Enzyme

Reactant.set_default_backend("tpu")

function simple_loss(x)
    return sum(x .^ 2)
end

function compute_grad(x)
    return Enzyme.gradient(Reverse, simple_loss, x)[1]
end

input_data = Reactant.to_rarray(Float32[1.0, 2.0, 3.0, 4.0])

grad_compiled = @compile compute_grad(input_data)
grads = grad_compiled(input_data)

println("Example 06: Automatic differentiation complete. Gradients:")
display(Array(grads))

