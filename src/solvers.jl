#=
Author: Cooper Simpson

SFN step solvers.
=#

using FastGaussQuadrature: gausslaguerre
using Krylov: CgLanczosShiftSolver, cg_lanczos_shift!, CraigmrSolver, craigmr!
using Arpack: eigs
using KrylovKit: eigsolve

#=
Find search direction using shifted CG Lanczos
=#
mutable struct KrylovSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    krylov_solver::CgLanczosShiftSolver #krylov inverse mat vec solver
    krylov_order::I #maximum Krylov subspace size
    quad_nodes::S #quadrature nodes
    quad_weights::S #quadrature weights
    p::S #search direction
end

function KrylovSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, quad_order::I=31, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

    #quadrature
    nodes, weights = gausslaguerre(quad_order, 0.0, reduced=true)

    if length(nodes) < quad_order
        quad_order = length(nodes)
        println("Quadrature weight precision reached, using $(quad_order) quadrature locations.")
    end

    #=
    NOTE: Performing some extra global operations here.
    - Integral constant
    - Rescaling weights
    - Squaring nodes
    =#
    @. weights = (2/pi)*weights*exp(nodes)
    @. nodes = nodes^2

    #krylov solver
    solver = CgLanczosShiftSolver(dim, dim, quad_order, type)
    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return KrylovSolver(solver, krylov_order, nodes, weights, type(undef, dim))
end

function step!(solver::KrylovSolver, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    solver.p .= 0
    
    shifts = solver.quad_nodes .+ λ

    cg_lanczos_shift!(solver.krylov_solver, Hv, b, shifts, itmax=solver.krylov_order, timemax=time_limit)

    @simd for i in eachindex(shifts)
        @inbounds solver.p .+= solver.quad_weights[i]*solver.krylov_solver.x[i]
    end

    return
end

#=
Find search direction using low-rank eigendecomposition with Arpack
=#
mutable struct KrylovKitSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    rank::I #
    p::S #search direction
end

function KrylovKitSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    # rank = Int(ceil(log(dim)))
    rank = Int(ceil(sqrt(dim)))
    
    return KrylovKitSolver(rank, type(undef, dim))
end

function step!(solver::KrylovKitSolver, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    solver.p .= 0

    D, V, info = eigsolve(Hv, solver.rank)

    @. D = inv(sqrt(D^2+λ))
    V = stack(V)
    mul!(solver.p, V*Diagonal(D)*V', b) #not the fastest way to do this

    return
end

#=
Find search direction using low-rank eigendecomposition with Arpack
=#
mutable struct ArpackSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    rank::I #
    p::S #search direction
end

function ArpackSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    # rank = Int(ceil(log(dim)))
    rank = Int(ceil(sqrt(dim)))

    return ArpackSolver(rank, type(undef, dim))
end

function step!(solver::ArpackSolver, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    solver.p .= 0

    D, V = eigs(Hv, nev=solver.rank, which=:LM, ritzvec=true)

    @. D = inv(sqrt(D^2+λ))
    # mul!(solver.p, V', b)
    # mul!(solver.p, Diagonal(D), solver.p)
    # mul!(solver.p, V, solver.p)
    mul!(solver.p, V*Diagonal(D)*V', b) #not the fastest way to do this

    return
end

#=
Find search direction using low-rank eigendecomposition with Arpack
=#
mutable struct EigenSolver{T<:AbstractFloat, S<:AbstractVector{T}}
    p::S #search direction
end

function EigenSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    return EigenSolver(type(undef, dim))
end

function step!(solver::EigenSolver, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    solver.p .= 0

    E = eigen(Hermitian(Matrix(Hv.op)))

    @. E.values = sqrt(E.values^2+λ)
    mul!(solver.p, inv(E), b)

    return
end