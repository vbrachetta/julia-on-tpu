#=
Example 02 - Matrix Multiplication (Float32)
Demonstrates host-to-device transfer via to_rarray, function
compilation via @compile, and retrieval of results from the TPU.
=#

using Reactant
using LinearAlgebra

Reactant.set_default_backend("tpu")

A_host = Float32[1 2 3; 4 5 6; 7 8 9]
B_host = Float32[9 8 7; 6 5 4; 3 2 1]

println("--- Original Matrices (CPU) ---")
display(A_host)
println()
display(B_host)

A_tpu = Reactant.to_rarray(A_host)
B_tpu = Reactant.to_rarray(B_host)

matmul(X, Y) = X * Y
matmul_compiled = @compile matmul(A_tpu, B_tpu)

result_tpu = matmul_compiled(A_tpu, B_tpu)
result_host = Array(result_tpu)

println("\n--- Result of A * B (Calculated on TPU) ---")
display(result_host)
println("\nExample 02: Matrix multiplication complete.")

