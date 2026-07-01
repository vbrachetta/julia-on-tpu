#=
Example 09 - Device-Side Random Number Generation
Compiles Julia's rand() call into the XLA graph so random numbers
are generated directly on the TPU without host-device transfer.
=#

using Reactant

Reactant.set_default_backend("tpu")

function generate_noise(shape_template)
    noise = rand(Float32, size(shape_template)...)
    return noise .* 0.1f0
end

template_tpu = Reactant.to_rarray(zeros(Float32, 100, 100))

noise_compiled = @compile generate_noise(template_tpu)
random_noise = noise_compiled(template_tpu)

println("Example 09: Device-side RNG complete. Sample value: ", Array(random_noise)[1,1])

