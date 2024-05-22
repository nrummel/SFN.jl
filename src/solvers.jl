#=
Author: Cooper Simpson

SFN step solvers.
=#

using FastGaussQuadrature: gausslaguerre, gausschebyshevt
using Krylov: CgLanczosShiftSolver, cg_lanczos_shift!, CgLanczosSolver, cg_lanczos!, CgLanczosShaleSolver, cg_lanczos_shale!
using KrylovKit: eigsolve, Lanczos, KrylovDefaults

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

    ζ = 0.5
    ξ = T(0.01)
    cg_atol = max(sqrt(eps(T)), min(ξ, ξ*λ^(1+ζ)))
    cg_rtol = max(sqrt(eps(T)), min(ξ, ξ*λ^(ζ)))

    cg_lanczos_shift!(solver.krylov_solver, Hv, b, shifts, itmax=solver.krylov_order, timemax=time_limit, atol=cg_atol, rtol=cg_rtol)

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

    # β = λ > 1 ? λ : one(λ) #NOTE: This could be set to some good estimate for where eigenvalues of Hv are clustered
    β = 1.0
    #trying to set β well
    # E = eigen(Hermitian(Matrix(Hv.op))) #NOTE: WORKS, Using Matrix function from LinearOperator.jl

    # Hv.nprod += length(b)

    # β = mean(abs.(E.values))
    # # # println("β: ", β)

    # @. E.values = E.values^2
    # M = pinv(E)
    # #

    shifts = (λ-β) .* solver.quad_nodes .+ (λ+β)
    scales = solver.quad_nodes .+ 1

    ζ = 0.5
    ξ = T(0.01)
    cg_atol = max(sqrt(eps(T)), min(ξ, ξ*λ^(1+ζ)))
    cg_rtol = max(sqrt(eps(T)), min(ξ, ξ*λ^(ζ)))

    cg_lanczos_shale!(solver.krylov_solver, Hv, b, shifts, scales, itmax=solver.krylov_order, timemax=time_limit, atol=cg_atol, rtol=cg_rtol)

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
    k = Int(ceil(rank*log(rank)))

    # krylov_solver = Lanczos(krylovdim=dim, maxiter=50, tol=100, orth=KrylovDefaults.orth, eager=false, verbosity=0)
    # return KrylovSolver(rank, krylov_solver, type(undef, dim))

    return KrylovSolver(rank, k, 10, rand(T, dim))
end

function step!(solver::KrylovSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    # D, V, info = eigsolve(Hv, rand(T, size(Hv,1)), solver.rank, :LM, solver.krylov_solver)
    D, V, info = eigsolve(Hv, rand(T, size(Hv,1)), solver.rank, :LM, krylovdim=solver.krylovdim, maxiter=solver.maxiter, tol=sqrt(λ))

    # push!(stats.krylov_iterations, info.numiter) #NOTE: This isn't right
    
    @. D = pinv(sqrt(D^2+λ))
    V = stack(V)

    cache = S(undef, length(D))

    mul!(cache, V', b)
    @. cache *= D
    mul!(solver.p, V, cache)

    return
end

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

    @. E.values = pinv(sqrt(E.values^2+λ))
    # mul!(solver.p, inv(E), b)

    cache = similar(b)

    mul!(cache, E.vectors', b)
    @. cache *= E.values
    mul!(solver.p, E.vectors, cache)

    return
end