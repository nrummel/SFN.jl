#=
Author: Cooper Simpson

CubicNewton optimization package.
=#
module RSFN

#=
Setup
=#
using LinearAlgebra

export RSFNOptimizer, minimize!

include("hvp.jl")
include("linesearch.jl")
include("optimizer.jl")

#=
If optional packages are loaded then export compatible functions.
=#
function __init__()
    
end

end #module
