using RegularizedMixedModels
using InteractiveUtils: versioninfo
using LinearAlgebra: BLAS
using Test

# there seem to be processor-specific issues and knowing this is helpful
println(versioninfo())
@static if VERSION ≥ v"1.7.0-DEV.620"
    println(BLAS.get_config())
else
    @show BLAS.vendor()
    if startswith(string(BLAS.vendor()), "openblas")
        println(BLAS.openblas_get_config())
    end
end

@testset "RegularizedLinearMixedModel" begin
    include("rlmm.jl")
end
