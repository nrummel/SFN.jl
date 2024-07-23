#=
Author: Cooper Simpson

SFN optimizer stats
=#

using Statistics: mean

export Stats

abstract type Stats{I, S1} end

mutable struct SFNStats{I<:Integer, S1<:Vector{<:AbstractFloat}} <: Stats{I,S1}
    converged::Bool #whether optimizer has converged
    iterations::I #number of optimizer iterations
    f_evals::I #number of function evaluations
    hvp_evals::I #number of hvp evaluations
    run_time::Float64 #iteration runtime
    f_seq::S1 #function value sequence
    g_seq::S1 #gradient norm sequence
    krylov_iterations::S1 #number of Krylov iterations #NOTE: We may not want this long term
    status::String #exit status
end

#=
Outer constructor

Input
=#
function Stats(type::Type{<:AbstractFloat})
    return SFNStats(false, 0, 0, 0, 0.0, type[], type[], type[], "")
end

function Base.show(io::IO, stats::Stats)
    print(io, "Converged: ", stats.converged, '\n',
                "Iterations: ", stats.iterations, '\n',
                "Function Evals: ", stats.f_evals, '\n',
                "Hvp Evals: ", stats.hvp_evals, '\n',
                "Run Time (s): ", stats.run_time, '\n',
                "Minimum: ", stats.f_seq[end], '\n',
                "Avg. Krylov Iterations: ", mean(stats.krylov_iterations), '\n',
                "Status: ", stats.status, '\n')
end

#=
Timer
https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/src/krylov_utils.jl
=#
elapsed(tic::UInt64) = (time_ns()-tic)/1e9

##
mutable struct ARCStats{I<:Integer, S1<:Vector{<:AbstractFloat}} <:Stats{I, S1}
    converged::Bool #whether optimizer has converged
    iterations::I #number of optimizer iterations
    f_evals::I #number of function evaluations
    hvp_evals::I #number of hvp evaluations
    run_time::Float64 #iteration runtime
    f_seq::S1 #function value sequence
    g_seq::S1 #gradient norm sequence
    krylov_iterations::S1 #number of Krylov iterations #NOTE: We may not want this long term
    status::String #exit status
    unsuccessful_iters::Int #exit status
end

#=
Outer constructor

Input
=#
function ARCStats(type::Type{<:AbstractFloat})
    return ARCStats(false, 0, 0, 0, 0.0, type[], type[], type[], "", 0)
end

function Base.show(io::IO, stats::ARCStats)
    print(io, "Converged: ", stats.converged, '\n',
                "Iterations: ", stats.iterations, '\n',
                "Function Evals: ", stats.f_evals, '\n',
                "Hvp Evals: ", stats.hvp_evals, '\n',
                "Run Time (s): ", stats.run_time, '\n',
                "Minimum: ", stats.f_seq[end], '\n',
                "Avg. Krylov Iterations: ", mean(stats.krylov_iterations), '\n',
                "Status: ", stats.status, '\n',
                "Unsuccessful iterations: ", stats.unsuccessful_iters, '\n',
                )
end