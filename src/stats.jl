#=
Author: Cooper Simpson

SFN optimizer stats
=#

mutable struct SFNStats{I<:Integer, S<:AbstractVector{<:AbstractFloat}}
    converged::Bool #whether optimizer has converged
    iterations::I #number of optimizer iterations
    hvp_evals::I #number of hvp evaluations
    f_seq::S #function value sequence
    g_seq::S #gradient norm sequence
end

#=
Outer constructor

Input
=#
function SFNStats(type::Type{<:AbstractFloat})
    return SFNStats(false, 0, 0, type[], type[])
end