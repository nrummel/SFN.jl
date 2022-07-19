#=
Author: Cooper Simpson

CubicNewton optimization package.
=#
module RSFN

#=
Setup
=#
using Requires
using LinearAlgebra

export RSFNOptimizer, minimize!

abstract type HvpOperator end

include("hvp_rdiff.jl")
include("hvp_zygote.jl")
include("logger.jl")
include("optimizer.jl")

#=
If Flux is loaded then export compatability functions.
=#
function __init__()
    @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        export StochasticRSFN
        include("flux.jl")
    end
end

end #module
