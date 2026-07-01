#=
Example 08 - 2D Stencil Computation (Laplacian)
Implements a 4-point Laplacian stencil on a 2D grid, which maps
directly to the XLA convolution primitive on TPU hardware.
=#

using Reactant

Reactant.set_default_backend("tpu")

function apply_stencil(A)
    inner = A[2:end-1, 2:end-1]
    up    = A[1:end-2, 2:end-1]
    down  = A[3:end,   2:end-1]
    left  = A[2:end-1, 1:end-2]
    right = A[2:end-1, 3:end]
    return up .+ down .+ left .+ right .- 4.0f0 .* inner
end

grid_tpu = Reactant.to_rarray(rand(Float32, 512, 512))

stencil_compiled = @compile apply_stencil(grid_tpu)
output = stencil_compiled(grid_tpu)

println("Example 08: Stencil computation complete. Output size: ", size(output))

