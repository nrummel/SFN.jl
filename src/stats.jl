#=
Author: Cooper Simpson

SFN optimizer stats
=#

mutable struct SFNStats
    converged::Bool #whether optimizer has converged
    iterations::Int #number of optimizer iterations
    hvp_evals::Int #number of hvp evaluations
    f_seq::AbstractVector{AbstractFloat} #function value sequence
    g_seq::AbstractVector{AbstractFloat} #gradient norm sequence
end

#=
Outer constructor

Input
=#
function SFNStats()
    return SFNStats(false, 0, 0, [], [])
end