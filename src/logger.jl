#=
Author: Cooper Simpson

Utilities for working with the cubic newton optimizer.
=#

#=
Tracks information
=#
mutable struct Logger
    hvp_calls::Real
end
