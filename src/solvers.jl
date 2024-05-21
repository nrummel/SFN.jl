#=
Author: Cooper Simpson

SFN step solvers.
=#

using FastGaussQuadrature: gausslaguerre, gausschebyshevt
using Krylov: CgLanczosShiftSolver, cg_lanczos_shift!, CgLanczosSolver, cg_lanczos!, CraigmrSolver, craigmr!, CgLanczosShaleSolver, cg_lanczos_shale!, hermitian_lanczos
using Arpack: eigs
using KrylovKit: eigsolve, Lanczos, KrylovDefaults
using RandomizedPreconditioners: NystromSketch, NystromPreconditioner

########################################################

#=
Find search direction using shifted CG Lanczos
=#
mutable struct RNSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    krylov_solver::CgLanczosShiftSolver #krylov inverse mat vec solver
    krylov_order::I #maximum Krylov subspace size
    p::S #search direction
end

function RNSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

    #krylov solver
    solver = CgLanczosShiftSolver(dim, dim, 1, type)
    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return RNSolver(solver, krylov_order, type(undef, dim))
end

function step!(solver::RNSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::Float64=Inf) where {T<:AbstractFloat, S<:AbstractVector, H<:HvpOperator}

    cg_lanczos_shift!(solver.krylov_solver, Hv, b, [sqrt(λ)], itmax=solver.krylov_order, timemax=time_limit)

    # if sum(solver.krylov_solver.converged) != length(shifts)
    #     println("WARNING: Solver failure")
    # end

    push!(stats.krylov_iterations, solver.krylov_solver.stats.niter)

    solver.p .= solver.krylov_solver.x[1]

    return
end

########################################################

#=
Find search direction using shifted CG Lanczos
=#
mutable struct GLKSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    krylov_solver::CgLanczosShiftSolver #krylov inverse mat vec solver
    krylov_order::I #maximum Krylov subspace size
    quad_nodes::S #quadrature nodes
    quad_weights::S #quadrature weights
    p::S #search direction
end

function GLKSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, quad_order::I=61, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

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

    return GLKSolver(solver, krylov_order, T.(nodes), T.(weights), type(undef, dim))
end

function step!(solver::GLKSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::Float64=Inf) where {T<:AbstractFloat, S<:AbstractVector, H<:HvpOperator}
    solver.p .= 0
    
    shifts = solver.quad_nodes .+ λ

    #perfect preconditioning
    # E = eigen(Hermitian(Matrix(Hv.op))) #NOTE: WORKS, Using Matrix function from LinearOperator.jl

    # Hv.nprod += length(b)

    # @. E.values = (E.values)^2

    # M = pinv(E)
    #

    cg_lanczos_shift!(solver.krylov_solver, Hv, b, shifts, itmax=solver.krylov_order, timemax=time_limit)

    # if sum(solver.krylov_solver.converged) != length(shifts)
    #     println("WARNING: Solver failure")
    # end

    push!(stats.krylov_iterations, solver.krylov_solver.stats.niter)

    @simd for i in eachindex(shifts)
        @inbounds solver.p .+= solver.quad_weights[i]*solver.krylov_solver.x[i]
    end

    return
end

########################################################

#=
Find search direction using shifted CG Lanczos
=#
mutable struct GCKSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    krylov_solver::CgLanczosShaleSolver #krylov inverse mat vec solver
    krylov_order::I #maximum Krylov subspace size
    quad_nodes::S #quadrature nodes
    quad_weights::S #quadrature weights
    p::S #search direction
end

function GCKSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, quad_order::I=61, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

    #quadrature
    nodes, weights = gausschebyshevt(quad_order)
    @. weights *= 2/pi #global scaling

    #krylov solver
    solver = CgLanczosShaleSolver(dim, dim, quad_order, type)
    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return GCKSolver(solver, krylov_order, nodes, weights, type(undef, dim))
end

function step!(solver::GCKSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    solver.p .= 0

    β = 1. #NOTE: This could be set to some good estimate for where eigenvalues of Hv are clustered

    #trying to set β well
    E = eigen(Hermitian(Matrix(Hv.op))) #NOTE: WORKS, Using Matrix function from LinearOperator.jl

    Hv.nprod += length(b)

    β = mean(abs.(E.values))
    # # # println("β: ", β)

    # @. E.values = E.values^2
    # M = pinv(E)
    # #

    shifts = (λ-β) .* solver.quad_nodes .+ (λ+β)
    scales = solver.quad_nodes .+ 1

    cg_lanczos_shale!(solver.krylov_solver, Hv, b, shifts, scales, itmax=solver.krylov_order, timemax=time_limit)

    # if sum(solver.krylov_solver.converged) != length(scales)
    #     println("WARNING: Solver failure")
    # end

    push!(stats.krylov_iterations, solver.krylov_solver.stats.niter)

    @simd for i in eachindex(scales)
        @inbounds solver.p .+= solver.quad_weights[i]*solver.krylov_solver.x[i]
    end

    @. solver.p *= sqrt(β)

    return
end

########################################################

#=
Find search direction using low-rank eigendecomposition with Arpack
=#
mutable struct KrylovSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    rank::I #
    # krylov_solver::Lanczos #
    krylovdim::I
    maxiter::I
    p::S #search direction
end

function KrylovSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    rank = Int(ceil(sqrt(dim)))
    k = Int(ceil(rank*1.5))

    # krylov_solver = Lanczos(krylovdim=dim, maxiter=50, tol=100, orth=KrylovDefaults.orth, eager=false, verbosity=0)
    # return KrylovSolver(rank, krylov_solver, type(undef, dim))

    return KrylovSolver(rank, dim, 10, rand(T, dim))
end

function step!(solver::KrylovSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    # D, V, info = eigsolve(Hv, rand(T, size(Hv,1)), solver.rank, :LM, solver.krylov_solver)
    D, V, info = eigsolve(Hv, rand(T, size(Hv,1)), solver.rank, :LM, krylovdim=solver.krylovdim, maxiter=solver.maxiter, tol=sqrt(λ))

    # push!(stats.krylov_iterations, info.numiter) #NOTE: This isn't right
    
    @. D = inv(sqrt(D^2+λ))
    V = stack(V)

    cache = S(undef, length(D))

    mul!(cache, V', b)
    @. cache *= D
    mul!(solver.p, V, cache)

    return
end

########################################################

# #=
# Find search direction using low-rank eigendecomposition with Arpack
# =#
# mutable struct KrylovTriSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
#     rank::I #
#     krylovdim::I
#     p::S #search direction
# end

# function KrylovTriSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
#     # rank = Int(ceil(log(dim)))
#     rank = Int(ceil(sqrt(dim)))
#     # rank = min(dim, 100)
#     k = Int(ceil(rank*1.5))
#     # k = dim

#     # krylov_solver = Lanczos(krylovdim=dim, maxiter=50, tol=100, orth=KrylovDefaults.orth, eager=false, verbosity=0)
    
#     # return KrylovSolver(rank, krylov_solver, type(undef, dim))
#     return KrylovTriSolver(rank, rank, rand(T, dim))
# end

# function step!(solver::KrylovTriSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
#     # solver.p .= 0

#     V, _, Tri = hermitian_lanczos(Hv, b, solver.krylovdim)
#     Tri = SymTridiagonal(Tri[1:solver.krylovdim,:])

#     E = eigen(Tri)
#     E = Eigen(E.values, V[:,1:solver.rank]*E.vectors)
#     @. E.values = sqrt(E.values^2+λ)
#     mul!(solver.p, inv(E), b) #not the fastest way to do this

#     # println(inv.(D[1:4]))

#     return
# end

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
mutable struct EigenSolver{T<:AbstractFloat, S<:AbstractVector{T}}
    p::S #search direction
end

function EigenSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    return EigenSolver(type(undef, dim))
end

function step!(solver::EigenSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    solver.p .= 0

    # E = eigen(Matrix(Hv)) #NOTE: FAILS OFTEN, Using custom Matrix function should be same as LinearOperator.jl
    E = eigen(Hermitian(Matrix(Hv.op))) #NOTE: WORKS, Using Matrix function from LinearOperator.jl

    Hv.nprod += length(b)

    @. E.values = sqrt(E.values^2+λ)
    mul!(solver.p, inv(E), b)

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