#=
Author: Cooper Simpson

CubicNewton optimization package.
=#
module SFN

#=
Setup
=#
using LinearAlgebra

export SFNOptimizer, minimize!

include("hvp.jl")
include("linesearch.jl")
include("optimizer.jl")

#=
If optional packages are loaded then export compatible functions.
=#
function __init__()
    
end

end #module