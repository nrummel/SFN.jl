#=
Author: Cooper Simpson

SFN optimizer stats
=#

mutable struct SFNStats{I<:Integer, S1<:AbstractVector{<:AbstractFloat}, S2<:AbstractVector{<:Integer}}
    converged::Bool #whether optimizer has converged
    iterations::I #number of optimizer iterations
    f_evals::I #number of function evaluations
    hvp_evals::I #number of hvp evaluations
    run_time::Float64 #iteration runtime
    f_seq::S1 #function value sequence
    g_seq::S1 #gradient norm sequence
    krylov_iterations::S2 #number of Krylov iterations #NOTE: We may not want this long term
end

#=
Outer constructor

Input
=#
function SFNStats(type::Type{<:AbstractFloat})
    return SFNStats(false, 0, 0, 0, 0.0, type[], type[], Int64[])
end

#=
Timer
https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/src/krylov_utils.jl
=#
elapsed(tic::UInt64) = (time_ns()-tic)/1e9