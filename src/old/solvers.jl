#=
Author: Cooper Simpson

SFN step solvers.
=#

using FastGaussQuadrature: gausslaguerre
using Krylov: CraigmrSolver, craigmr!
using Arpack: eigs
using RandomizedPreconditioners: NystromSketch, NystromPreconditioner
using IterativeSolvers: lobpcg!, LOBPCGIterator, lobpcg

########################################################

#=
Find search direction using 
=#
mutable struct LOBPCGSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    rank::I
    maxiter::I
    # iterator::LOBPCGSolver
    p::S #search direction
end

function LOBPCGSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    rank = Int(ceil(sqrt(dim)))
    k = Int(ceil(rank*log(rank)))

    # iterator = LOBPCGIterator()

    return LOBPCGSolver(rank, k, type(undef, dim))
end

function step!(solver::LOBPCGSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    
    # r = lobpcg!(solver.iterator, maxiter=solver.maxiter)
    r = lobpcg(Hv, true, solver.rank, maxiter=solver.maxiter)
    
    @. r.λ = inv(sqrt((r.λ)^2+λ))

    cache = S(undef, length(D))

    mul!(cache, r.X', b)
    @. cache *= r.λ
    mul!(solver.p, r.X, cache)

    return
end

########################################################

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

function step!(solver::ArpackSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    solver.p .= 0

    D, V = eigs(Hv, nev=solver.rank, which=:LM, ritzvec=true)

    @. D = pinv(sqrt(D^2+λ))
    # mul!(solver.p, V', b)
    # mul!(solver.p, Diagonal(D), solver.p)
    # mul!(solver.p, V, solver.p)
    mul!(solver.p, V*Diagonal(D)*V', b) #not the fastest way to do this

    return
end

########################################################

#=
Find search direction using full eigendecomposition
=#
mutable struct NystromPCGSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    k::I #rank
    r::I #sketch size
    krylov_solver::CgLanczosShiftSolver #krylov inverse mat vec solver
    krylov_order::I #maximum Krylov subspace size
    quad_nodes::S #quadrature nodes
    quad_weights::S #quadrature weights
    p::S #search direction
end

function NystromPCGSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, quad_order::I=61, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}
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

    #nystrom rank
    k = Int(ceil(log(dim)))
    r = Int(ceil(1.5*k))

    return NystromPCGSolver(k, r, solver, krylov_order, nodes, weights, type(undef, dim))
end

function step!(solver::NystromPCGSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    solver.p .= 0

    shifts = solver.quad_nodes .+ λ

    # A = Matrix(Hv.op)
    # A = A*A

    # println(A)

    # P = nystrom(A, solver.k, solver.r, λ)
    Pinv = NystromPreconditionerInverse(NystromSketch(Hv, solver.k, solver.r), 0)

    # println(Matrix(Pinv))

    cg_lanczos_shift!(solver.krylov_solver, Hv, b, shifts, M=Pinv, ldiv=false, itmax=solver.krylov_order, timemax=time_limit)

    # if sum(solver.krylov_solver.converged) != length(shifts)
    #     println("WARNING: Solver failure")
    # end

    push!(stats.krylov_iterations, solver.krylov_solver.stats.niter)

    @simd for i in eachindex(solver.quad_nodes)
        @inbounds solver.p .+= solver.quad_weights[i]*solver.krylov_solver.x[i]
    end

    return
end

########################################################

#=
Find search direction using CRAIGMR
=#
mutable struct CraigSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    krylov_solver::CraigmrSolver #krylov inverse mat vec solver
    krylov_order::I #maximum Krylov subspace size
    quad_nodes::S #quadrature nodes
    quad_weights::S #quadrature weights
    p::S #search direction
end

function CraigSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, quad_order::I=61, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

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
    solver = CraigmrSolver(dim, dim, type)
    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return CraigSolver(solver, krylov_order, nodes, weights, type(undef, dim))
end

function step!(solver::CraigSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    solver.p .= 0
    
    shifts = sqrt.(solver.quad_nodes .+ λ)

    solved = true

    @inbounds for i in eachindex(shifts)
        craigmr!(solver.krylov_solver, Hv, b, λ=shifts[i], itmax=solver.krylov_order, timemax=time_limit)

        solved = solved && solver.krylov_solver.stats.solved

        solver.p .+= solver.quad_weights[i]*solver.krylov_solver.y
    end

    # if solved == false
    #     println("WARNING: Solver failure")
    # end

    return
end

########################################################

#=
Find search direction using randomized definite Nystrom
=#
mutable struct NystromDefiniteSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    k::I #rank
    r::I #sketch size
    p::S #search direction
end

function NystromDefiniteSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, k::I=Int(ceil(sqrt(dim))), r::I=Int(ceil(1.5*k))) where {I<:Integer, T<:AbstractFloat}
    return NystromDefiniteSolver(k, r, type(undef, dim))
end

function step!(solver::NystromDefiniteSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    P = NystromPreconditioner(NystromSketch(Hv, solver.k), λ)

    mul!(P.cache, P.A_nys.U', b)
    @. P.cache *= (P.λ + P.μ) * 1 / sqrt(P.A_nys.Λ.diag + P.μ) - 1
    mul!(solver.p, P.A_nys.U, P.cache)
    @. solver.p = b + solver.p

    return
end

########################################################

#=
Find search direction using randomized indefinite Nystrom
=#
mutable struct NystromIndefiniteSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    r::I #rank
    s::I #sketch size
    p::S #search direction
end

function NystromIndefiniteSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, r::I=Int(ceil(sqrt(dim))), c::T=1.5) where {I<:Integer, T<:AbstractFloat}
    # return NystromIndefiniteSolver(r, Int(ceil(c*r)), type(undef, dim))
    return NystromIndefiniteSolver(20, 50, type(undef, dim))
end

function step!(solver::NystromIndefiniteSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    Ω = randn(size(Hv,1), solver.s) #Guassian test matrix

    C = Hv*Ω #sketch
    W = Ω'*C #sketch
    Λ, V = eigen(Hermitian(W), sortby=(-)∘(abs)) #eigendecomposition
    Λ, V = Λ[1:solver.r], V[:,1:solver.r]

    Wr = V*Diagonal(pinv.(Λ))*V'

    Q, R = qr(C)
    Σ, U = eigen(Hermitian(R*Wr*R'), sortby=(-)∘(abs))
    E = Eigen(Σ[1:solver.r], Q*U[:,1:solver.r])
    @. E.values = sqrt(E.values^2+λ)

    mul!(solver.p, pinv(E), b)

    # println(pinv.(E.values[1:4]))

    return 
end

########################################################

#=
Find search direction using randomized SVD
=#
mutable struct RSVDSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    k::I #rank
    r::I #sketch size
    p::S #search direction
end

function RSVDSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, k::I=Int(ceil(sqrt(dim))), r::I=Int(ceil(1.5*k))) where {I<:Integer, T<:AbstractFloat}
    return RSVDSolver(k, r, type(undef, dim))
end

function step!(solver::RSVDSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    # Ω = srft(solver.k)
    Ω = randn(size(Hv,1), solver.k)
    
    Y=Hv*Ω; Q = Matrix(qr!(Y).Q)
    B=(Q'Y)\(Q'Ω)
    E=eigen!(B)
    E=Eigen(E.values, Q*real.(E.vectors))

    @. E.values = sqrt(real(E.values)^2+λ)
    mul!(solver.p, pinv(E), b)

    return
end