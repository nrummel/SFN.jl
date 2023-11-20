#=
Author: Cooper Simpson

SFN optimizer.
=#

using FastGaussQuadrature: gausslaguerre
using Krylov: CgLanczosShiftSolver, cg_lanczos_shift!
using Zygote: pullback

#=
SFN optimizer struct.
=#
mutable struct SFNOptimizer{T1<:Real, T2<:AbstractFloat, S<:AbstractVector{T2}, I<:Integer, LS}
    M::T1 #hessian lipschitz constant
    ϵ::T2 #regularization minimum
    linesearch::LS #whether to use linsearch
    quad_nodes::S #quadrature nodes
    quad_weights::S #quadrature weights
    krylov_solver::CgLanczosShiftSolver #krylov inverse mat vec solver
    krylov_order::I #maximum Krylov subspace size
    atol::T2 #absolute gradient norm tolerance
    rtol::T2 #relative gradient norm tolerance
end

#=
Outer constructor

NOTE: FGQ cant currently handle anything other than Float64

Input:
    dim :: dimension of parameters
    type :: parameter type
    M :: hessian lipschitz constant
    ϵ :: regularization minimum
    quad_order :: number of quadrature nodes
    krylov_order :: max Krylov subspace size
    tol :: gradient norm tolerance to declare convergence
=#
function SFNOptimizer(dim::I, type::Type{<:AbstractVector{T2}}=Vector{Float64}; M::T1=1, ϵ::T2=eps(Float64), linesearch::Bool=false, quad_order::I=20, krylov_order::I=dim, atol::T2=1e-6, rtol::T2=1e-6) where {T1<:Real, T2<:AbstractFloat, I<:Integer}
    #regularization
    if linesearch
        M = 1.0
        linesearch = SFNLineSearcher()
    else
        linesearch = nothing
    end
    
    #quadrature
    nodes, weights = gausslaguerre(quad_order, 0.0, reduced=true)

    if size(nodes, 1) < quad_order
        quad_order = size(nodes, 1)
        println("Quadrature weight precision reached, using $(size(nodes,1)) quadrature locations.")
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

    return SFNOptimizer(M, ϵ, linesearch, nodes, weights, solver, krylov_order, atol, rtol)
end

#=
Repeatedly applies the SFN iteration to minimize the function.

Input:
    opt :: SFNOptimizer
    x :: initialization
    f :: scalar valued function
    itmax :: maximum iterations
    time_limit :: maximum run time
=#
function minimize!(opt::SFNOptimizer, x::S, f::F; itmax::I=1000, time_limit::T2=Inf) where {T1<:AbstractFloat, S<:AbstractVector{T1}, T2, F, I}
    #setup hvp operator
    Hv = RHvpOperator(f, x)

    #
    function fg!(grads::S, x::S)
        
        fval, back = let f=f; pullback(f, x) end
        grads .= back(one(fval))[1]

        return fval
    end

    #iterate
    stats = iterate!(opt, x, f, fg!, Hv, itmax, time_limit)

    return stats
end

#=
Repeatedly applies the SFN iteration to minimize the function.

Input:
    opt :: SFNOptimizer
    x :: initialization
    f :: scalar valued function
    g! :: inplace gradient function of f
    H :: hvp generator
    itmax :: maximum iterations
    time_limit :: maximum run time
=#
function minimize!(opt::SFNOptimizer, x::S, f::F1, fg!::F2, H::L; itmax::I=1000, time_limit::T=Inf) where {T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, L, I}
    #setup hvp operator
    Hv = LHvpOperator(H, x)

    #iterate
    stats = iterate!(opt, x, f, fg!, Hv, itmax, time_limit)

    return stats
end

#=
Repeatedly applies the SFN iteration to minimize the function.

Input:
    opt :: SFNOptimizer
    x :: initialization
    f :: scalar valued function
    fg! :: compute f and gradient norm after inplace update of gradient
    Hv :: hvp operator
    itmax :: maximum iterations
    time_limit :: maximum run time
=#
function iterate!(opt::SFNOptimizer, x::S, f::F1, fg!::F2, Hv::H, itmax::I, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, H<:HvpOperator, I}
    #start time
    tic = time_ns()
    
    #stats
    stats = SFNStats(T)
    converged = false
    iterations = 0
    
    #gradient allocation
    grads = similar(x)

    #compute function and gradient
    fval = fg!(grads, x)
    g_norm = norm(grads)

    #compute tolerance
    tol = opt.atol + opt.rtol*g_norm

    #initial stats
    push!(stats.f_seq, fval)
    push!(stats.g_seq, g_norm)

    #iterate
    while iterations<itmax+1
        #check gradient norm
        if g_norm <= tol
            converged = true
            break
        end

        #check other exit conditions
        time = elapsed(tic)

        if (time>=time_limit) || (iterations==itmax)
            break
        end

        #step
        step!(opt, stats, x, f, grads, Hv, fval, g_norm, time_limit-time)

        #update function and gradient
        fval = fg!(grads, x)
        g_norm = norm(grads)

        #update stats
        push!(stats.f_seq, fval)
        push!(stats.g_seq, g_norm)

        #update hvp operator
        update!(Hv, x)

        #increment
        iterations += 1
    end

    #update stats
    stats.converged = converged
    stats.iterations = iterations
    stats.f_evals += iterations+1
    stats.hvp_evals = Hv.nProd
    stats.run_time = elapsed(tic)

    return stats
end


#=
Computes an update step according to the shifted Lanczos-CG update rule.

Input:
    opt :: SFNOptimizer
    x :: current iterate
    f :: scalar valued function
    grads :: function gradients
    Hv :: hessian operator
    g_norm :: gradient norm
    linesearch:: whether to use linesearch
=#
function step!(opt::SFNOptimizer, stats::SFNStats, x::S, f::F, grads::S, Hv::H, fval::T, g_norm::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, F, H<:HvpOperator}
    #compute regularization
    λ = opt.M*g_norm

    #compute shifts
    shifts = opt.quad_nodes .+ λ

    #compute CG Lanczos quadrature integrand ((tᵢ²+λₖ)I+Hₖ²)⁻¹gₖ
    cg_lanczos_shift!(opt.krylov_solver, Hv, grads, shifts, itmax=opt.krylov_order, timemax=time_limit)

    #
    push!(stats.krylov_iterations, opt.krylov_solver.stats.niter)

    #evaluate integral and update
    if isnothing(opt.linesearch)
        @simd for i in eachindex(shifts)
            @inbounds x .-= opt.quad_weights[i]*opt.krylov_solver.x[i]
        end
    else
        p = zero(x) #NOTE: Can we not allocate new space for this somehow?

        @simd for i in eachindex(shifts)
            @inbounds p .-= opt.quad_weights[i]*opt.krylov_solver.x[i]
        end

        search!(opt.linesearch, stats, x, p, f, fval, λ)
    end

    # println(opt.krylov_solver.stats.status)

    return nothing
end
