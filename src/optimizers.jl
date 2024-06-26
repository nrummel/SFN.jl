#=
Author: Cooper Simpson

Newton-type optimizers.
=#

abstract type Optimizer end

########################################################

#=
SFN optimizer struct.
=#
mutable struct SFNOptimizer{T1<:Real, T2<:AbstractFloat, S} <: Optimizer
    M::T1 #hessian lipschitz constant
    ϵ::T2 #regularization minimum
    solver::S #search direction solver
    linesearch::Bool #whether to use linsearch
    η::T2 #step-size
    α::T2 #linesearch factor
    atol::T2 #absolute gradient norm tolerance
    rtol::T2 #relative gradient norm tolerance
end

#=
Outer constructor

NOTE: FGQ cant currently handle anything other than Float64

Input:
    dim :: dimension of parameters
    solver :: search direction solver
    M :: hessian lipschitz constant
    ϵ :: regularization minimum
    linsearch :: whether to use linesearch
    η :: step-size in (0,1)
    α :: linsearch factor in (0,1)
    atol :: absolute gradient norm tolerance
    rtol :: relative gradient norm tolerance
=#
function SFNOptimizer(dim::I, solver::Symbol=:KrylovSolver; M::T1=1e-8, ϵ::T2=eps(Float64), linesearch::Bool=false, η::T2=1.0, α::T2=0.5, atol::T2=1e-5, rtol::T2=1e-6) where {I<:Integer, T1<:Real, T2<:AbstractFloat}
    #regularization
    @assert (0≤M && 0≤ϵ)

    if linesearch
        @assert (0<α && α<1)
        @assert (0<η && η≤1)
    end

    solver = eval(solver)(dim)

    return SFNOptimizer(M, ϵ, solver, linesearch, η, α, atol, rtol)
end

########################################################

#=
ARC optimizer struct.
=#
mutable struct ARCOptimizer{T<:AbstractFloat, S}
    solver::S #search direction solver
    linesearch::Bool #
    atol::T #absolute gradient norm tolerance
    rtol::T #relative gradient norm tolerance
end

#=
Outer constructor

NOTE: FGQ cant currently handle anything other than Float64

Input:
    dim :: dimension of parameters
    solver :: search direction solver
    M :: hessian lipschitz constant
    ϵ :: regularization minimum
    linsearch :: whether to use linesearch
    η :: step-size in (0,1)
    α :: linsearch factor in (0,1)
    atol :: absolute gradient norm tolerance
    rtol :: relative gradient norm tolerance
=#
function ARCOptimizer(dim::I; atol::T=1e-5, rtol::T=1e-6) where {I<:Integer, T<:AbstractFloat}

    solver = ARCSolver(dim)

    return ARCOptimizer(solver, true, atol, rtol)
end
