#=
Author: Cooper Simpson

CubicNewton optimization package.
=#
module CubicNewton

#=
Setup
=#
using Requires
using Zygote: gradient, withgradient
using ForwardDiff: partials, Dual

export CubicNewtonOpt

include("utilities.jl")
include("optimizers.jl")

#=
If Flux is loaded then export compatability functions.
=#
function __init__()
    @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        export train!

        include("flux_interface.jl")
    end
end

end #module
