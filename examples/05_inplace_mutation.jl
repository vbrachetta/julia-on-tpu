#=
Example 05 - In-Place Buffer Mutation
Validates in-place updates via .= without new memory allocations,
a pattern used in SGD weight updates to avoid OOM errors.
=#

using Reactant

Reactant.set_default_backend("tpu")

function apply_update!(weights, grads, lr)
    weights .-= lr .* grads
    return nothing
end

w_tpu = Reactant.to_rarray(fill(1.0f0, 5, 5))
g_tpu = Reactant.to_rarray(fill(0.1f0, 5, 5))
learning_rate = 0.01f0

update_kernel! = @compile apply_update!(w_tpu, g_tpu, learning_rate)
update_kernel!(w_tpu, g_tpu, learning_rate)

println("Example 05: In-place mutation complete. Updated weights:")
display(Array(w_tpu))

