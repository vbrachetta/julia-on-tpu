#=
Example 01 - Backend Initialisation and Basic Compiled Addition
Verifies that Reactant.jl can target the TPU backend and compile
a simple element-wise addition via the XLA compiler.
=#

using Reactant

Reactant.set_default_backend("tpu")

add(x, y) = x .+ y
x = rand(Float32, 100, 100)
y = rand(Float32, 100, 100)

compiled_add = Reactant.compile(add, (x, y))
result = compiled_add(x, y)

println("Example 01: Success. Shape: ", size(result))

