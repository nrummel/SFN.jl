#=
Author: Cooper Simpson

Utilities for working with the cubic newton optimizer.
=#

#=
Tracks information
=#
Base.@kwdef mutable struct Logger{T<:Int}
    fcalls::T = 0
    gcalls::T = 0
    hcalls::T = 0
end
