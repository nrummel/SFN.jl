#=
Author: Cooper Simpson

Utilities for working with the cubic newton optimizer.
=#

#=
Tracks information
=#
Base.@kwdef mutable struct Logger
    hvps::Int64 = 0
end
