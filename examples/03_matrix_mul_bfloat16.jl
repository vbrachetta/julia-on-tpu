#=
Example 03 - Matrix Multiplication (BFloat16)
Repeats the Float32 matrix multiplication using BFloat16, the
native precision of TPU v4 matrix units.
=#

using Reactant, BFloat16s

Reactant.set_default_backend("tpu")

function my_matmul(X, Y)
    return X * Y
end

A_host = BFloat16[1.5 2.5; 3.5 4.5]
B_host = BFloat16[0.5 1.0; 2.0 2.5]

A_tpu = Reactant.to_rarray(A_host)
B_tpu = Reactant.to_rarray(B_host)

matmul_compiled = @compile my_matmul(A_tpu, B_tpu)
result_tpu = matmul_compiled(A_tpu, B_tpu)

println("Example 03: Result in BFloat16:")
display(Array(result_tpu))
