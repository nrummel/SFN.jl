#=
Author: Cooper Simpson

SFN step solvers.
=#

using FastGaussQuadrature: gausslaguerre, gausschebyshevt
using Krylov: CgLanczosShiftSolver, cg_lanczos_shift!, CgLanczosSolver, cg_lanczos!, CgLanczosShaleSolver, cg_lanczos_shale!
using KrylovKit: eigsolve, Lanczos, KrylovDefaults
using IterativeSolvers: powm!

########################################################

#=
Shifted CG Lanczos with Gauss-Laguerre quadrature.
=#
mutable struct GLKSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    krylov_solver::CgLanczosShiftSolver #Krylov solver
    krylov_order::I #maximum Krylov subspace size
    quad_nodes::S #quadrature nodes
    quad_weights::S #quadrature weights
    p::S #search direction
end

function GLKSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, quad_order::I=61, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

    #Quadrature
    nodes, weights = gausslaguerre(quad_order, 0.0, reduced=true)

    if length(nodes) < quad_order
        quad_order = length(nodes)
        println("Quadrature weight precision reached, using $(quad_order) quadrature locations.")
    end

    #=
    Global operations
    - Integral constant
    - Rescaling weights
    - Squaring nodes
    =#
    @. weights = (2/pi)*weights*exp(nodes)
    @. nodes = nodes^2

    #Krylov solver
    solver = CgLanczosShiftSolver(dim, dim, quad_order, type)
    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return GLKSolver(solver, krylov_order, T.(nodes), T.(weights), type(undef, dim))
end

function step!(solver::GLKSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::Float64=Inf) where {T<:AbstractFloat, S<:AbstractVector, H<:HvpOperator}
    
    #Reset search direction
    solver.p .= 0

    #Quadrature scaling factor
    # E = eigen(Matrix(Hv)) #NOTE: WORKS, Using Matrix function from LinearOperator.jl

    # c1 = maximum(abs, E.values) #+ rand([-1,1])*1e2
    c = eigmax(Hv)
    # println(abs(c1-c))

    #Shifts
    shifts = c^2*solver.quad_nodes .+ λ
    
    #Tolerance
    ζ = 0.5
    ξ = T(0.01)
    cg_atol = max(sqrt(eps(T)), min(ξ, ξ*λ^(1+ζ)))
    cg_rtol = max(sqrt(eps(T)), min(ξ, ξ*λ^(ζ)))

    #CG solves
    cg_lanczos_shift!(solver.krylov_solver, Hv, b, shifts, itmax=solver.krylov_order, timemax=time_limit, atol=1e-6, rtol=1e-6)

    if sum(solver.krylov_solver.converged) != length(shifts)
        println("WARNING: Solver failure")
    end

    push!(stats.krylov_iterations, solver.krylov_solver.stats.niter)

    #Update search direction
    for i in eachindex(shifts)
        @inbounds solver.p .+= c*solver.quad_weights[i]*solver.krylov_solver.x[i] #NOTE: Should we multiply by c outside of loop?
    end

    return
end

########################################################

#=
Shifted and scaled CG Lanczos with Gauss-Chebyshev quadrature.
=#
mutable struct GCKSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    krylov_solver::CgLanczosShaleSolver #Krylov solver
    krylov_order::I #maximum Krylov subspace size
    quad_nodes::S #quadrature nodes
    quad_weights::S #quadrature weights
    p::S #search direction
end

function GCKSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, quad_order::I=61, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

    #Quadrature
    nodes, weights = gausschebyshevt(quad_order)
    @. weights *= 2/pi #global scaling

    #Krylov solver
    solver = CgLanczosShaleSolver(dim, dim, quad_order, type)
    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return GCKSolver(solver, krylov_order, nodes, weights, type(undef, dim))
end

function step!(solver::GCKSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    
    #Reset search direction
    solver.p .= 0

    #Quadrature constant
    #=
    This could be set to some good estimate for where eigenvalues of Hv are clustered.
    - Trace
    - Maximum
    - Median
    =#
    β = 1.0
    # β = λ > 1 ? λ : one(λ) 
    # E = eigen(Matrix(Hv)) #NOTE: WORKS, Using Matrix function from LinearOperator.jl
    # Hv.nprod += length(b)
    # β = mean(abs.(E.values))

    #Shifts and scalings
    shifts = (λ-β) .* solver.quad_nodes .+ (λ+β)
    scales = solver.quad_nodes .+ 1

    #Tolerance
    ζ = 0.5
    ξ = T(0.01)
    cg_atol = max(sqrt(eps(T)), min(ξ, ξ*λ^(1+ζ)))
    cg_rtol = max(sqrt(eps(T)), min(ξ, ξ*λ^(ζ)))

    #CG Solves
    cg_lanczos_shale!(solver.krylov_solver, Hv, b, shifts, scales, itmax=solver.krylov_order, timemax=time_limit, atol=cg_atol, rtol=cg_rtol)

    if sum(solver.krylov_solver.converged) != length(scales)
        println("WARNING: Solver failure")
    end

    push!(stats.krylov_iterations, solver.krylov_solver.stats.niter)

    #Update search direction
    for i in eachindex(scales)
        @inbounds solver.p .+= solver.quad_weights[i]*solver.krylov_solver.x[i]
    end

    @. solver.p *= sqrt(β)

    return
end

########################################################

#=
Krylov based low-rank approximation.
=#
mutable struct KrylovSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    rank::I #target rank
    # krylov_solver::Lanczos #Krylov solver
    krylovdim::I #maximum Krylov subspace size
    maxiter::I #maximum restarts
    p::S #search direction
end

function KrylovSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    rank = Int(ceil(sqrt(dim)))
    # k = Int(ceil(rank*log(rank)))
    k = dim

    # krylov_solver = Lanczos(krylovdim=dim, maxiter=50, tol=100, orth=KrylovDefaults.orth, eager=false, verbosity=0)
    # return KrylovSolver(rank, krylov_solver, type(undef, dim))

    return KrylovSolver(rank, k, 1, rand(T, dim))
end

function step!(solver::KrylovSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Low-rank eigendecomposition
    # D, V, info = eigsolve(Hv, rand(T, size(Hv,1)), solver.rank, :LM, solver.krylov_solver)
    D, V, info = eigsolve(Hv, rand(T, size(Hv,1)), solver.rank, :LM, krylovdim=solver.krylovdim, maxiter=solver.maxiter, tol=1e-1)

    push!(stats.krylov_iterations, info.numops)
    
    #Select, collect, and update eigenstuff
    # idx = min(info.converged, solver.rank)

    # D = D[1:info.converged]
    # V = V[1:info.converged]

    @. D = pinv(sqrt(D^2+λ))
    V = stack(V)

    #Temporary memory
    #NOTE: These could be part of the solver struct
    cache1 = S(undef, length(D))
    cache2 = similar(b)

    #Update search direction
    mul!(cache1, V', b)
    mul!(cache2, V, cache1)
    @. cache1 *= D
    mul!(solver.p, V, cache1)
    @. solver.p += inv(sqrt(λ))*(b-cache2)

    return
end

########################################################

#=
Full eigendecomposition.
=#
mutable struct EigenSolver{T<:AbstractFloat, S<:AbstractVector{T}}
    p::S #search direction
end

function EigenSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    return EigenSolver(type(undef, dim))
end

function step!(solver::EigenSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Eigendecomposition
    E = eigen(Matrix(Hv))

    Hv.nprod += length(b)

    #Update eigenvalues
    @. E.values = pinv(sqrt(E.values^2+λ))

    #Temporary memory
    cache = similar(b)

    #Update search direction
    mul!(cache, E.vectors', b)
    @. cache *= E.values
    mul!(solver.p, E.vectors, cache)

    return
end

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

    λ = sqrt(λ)

    ζ = 0.5
    ξ = T(0.01)
    cg_atol = max(sqrt(eps(T)), min(ξ, ξ*λ^(1+ζ)))
    cg_rtol = max(sqrt(eps(T)), min(ξ, ξ*λ^(ζ)))
    
    cg_lanczos_shift!(solver.krylov_solver, Hv, b, [λ], itmax=solver.krylov_order, timemax=time_limit, atol=cg_atol, rtol=cg_rtol)

    # if sum(solver.krylov_solver.converged) != length(shifts)
    #     println("WARNING: Solver failure")
    # end

    push!(stats.krylov_iterations, solver.krylov_solver.stats.niter)

    solver.p .= solver.krylov_solver.x[1]

    return
end