#=
Author: Cooper Simpson

CubicNewton optimization package.
=#
module CubicNewton

#=
Setup
=#
using Requires
using LinearAlgebra
using Zygote: gradient, withgradient
using ForwardDiff: partials, Dual
using Krylov: cg_lanczos

export ShiftedLanczosCG, Eigen, minimize!

include("utilities.jl")
include("optimizers.jl")

#=
If Flux is loaded then export compatability functions.
=#
function __init__()
    @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        export StochasticCubicNewton

        include("flux.jl")
    end
end

end #module
