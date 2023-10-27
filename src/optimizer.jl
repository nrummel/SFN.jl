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
mutable struct SFNOptimizer{T1<:Real, T2<:AbstractFloat, S<:AbstractVector{T2}, I<:Integer}
    M::T1 #hessian lipschitz constant
    ϵ::T2#regularization minimum
    quad_nodes::S #quadrature nodes
    quad_weights::S #quadrature weights
    krylov_solver::CgLanczosShiftSolver #krylov inverse mat vec solver
    krylov_itmax::I #maximum Krylov subspace size
    tol::T2 #gradient norm tolerance for exit
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
=#
function SFNOptimizer(dim::I, type::Type{<:AbstractVector{T2}}=Vector{Float64}; M::T1=1, ϵ::T2=eps(Float64), quad_order::I=20, krylov_order::I=0, tol::T2=1e-6) where {T1<:Real, T2<:AbstractFloat, I<:Integer}
    #quadrature
    nodes, weights = gausslaguerre(quad_order, 0.0, reduced=true)

    if size(nodes, 1) < quad_order
        quad_order = size(nodes, 1)
        println("Quadrature weight precision reached, using $(size(nodes,1)) quadrature locations.")
    end

    #krylov solver
    solver = CgLanczosShiftSolver(dim, dim, quad_order, type)

    #=
    NOTE: Performing some extra global operations here.
    - Integral constant
    - Rescaling weights
    - Squaring nodes
    =#
    @. weights = (2/pi)*weights*exp(nodes)
    @. nodes = nodes^2

    #max number of Krylov iterations
    if krylov_order == 0
        # itmax = round(Int, sqrt(dim))
        krylov_itmax = 2*dim
    else
        krylov_itmax = krylov_order
    end

    return SFNOptimizer(M, ϵ, nodes, weights, solver, krylov_itmax, tol)
end

#=
Repeatedly applies the SFN iteration to minimize the function.

Input:
    opt :: SFNOptimizer
    x :: initialization
    f :: scalar valued function
    itmax :: maximum iterations
    linesearch :: whether to use step-size with linesearch
=#
function minimize!(opt::SFNOptimizer, x::S, f::F; itmax::I=1000, linesearch::Bool=false) where {T<:AbstractFloat, S<:AbstractVector{T}, F, I<:Integer}
    #setup hvp operator
    Hv = RHvpOperator(f, x)

    #
    function fg!(grads::S, x::S)
        
        fval, back = let f=f; pullback(f, x) end
        grads .= back(one(fval))[1]

        return fval
    end

    #iterate
    tic = time_ns
    stats = iterate!(opt, x, f, fg!, Hv, itmax, linesearch)
    toc = elapsed(tic)

    stats.run_time = toc

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
    linesearch :: whether to use step-size with linesearch
=#
function minimize!(opt::SFNOptimizer, x::S, f::F1, fg!::F2, H::L; itmax::I=1000, linesearch::Bool=false) where {T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, L, I<:Integer}
    #setup hvp operator
    Hv = LHvpOperator(H, x)

    #iterate
    tic = time_ns()
    stats = iterate!(opt, x, f, fg!, Hv, itmax, linesearch)
    toc = elapsed(tic)

    stats.run_time = toc

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
    linesearch :: whether to use step-size with linesearch
=#
function iterate!(opt::SFNOptimizer, x::S, f::F1, fg!::F2, Hv::H, itmax::I, linesearch::Bool) where {T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, H<:HvpOperator, I<:Integer}
    stats = SFNStats(T)
    
    grads = similar(x)

    converged = false
    iterations = itmax
    
    for i = 1:itmax
        #compute function and gradient
        fval = fg!(grads, x)
        g_norm = norm(grads)

        #update stats
        push!(stats.f_seq, fval)
        push!(stats.g_seq, g_norm)

        #step
        step!(opt, x, f, grads, Hv, fval, g_norm, linesearch)

        #check gradient norm
        if g_norm <= opt.tol
            converged = true
            iterations = i
            break
        end

        #update hvp operator
        update!(Hv, x)
    end

    #update stats
    stats.converged = converged
    stats.iterations = iterations
    stats.hvp_evals = Hv.nProd

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
function step!(opt::SFNOptimizer, x::S, f::F, grads::S, Hv::H, fval::T, g_norm::T, linesearch::Bool=false) where {T<:AbstractFloat, S<:AbstractVector{T}, F, H<:HvpOperator}
    #compute regularization
    λ = opt.M*g_norm

    #compute shifts
    shifts = opt.quad_nodes .+ λ

    #compute CG Lanczos quadrature integrand ((tᵢ²+λₖ)I+Hₖ²)⁻¹gₖ
    cg_lanczos_shift!(opt.krylov_solver, Hv, grads, shifts, itmax=opt.krylov_itmax)

    #evaluate integral and update
    if linesearch
        p = zero(x) #NOTE: Can we not allocate new space for this somehow?

        @simd for i in eachindex(shifts)
            @inbounds p .-= opt.quad_weights[i]*opt.krylov_solver.x[i]
        end

        search!(x, p, f, fval, λ)
    else
        @simd for i in eachindex(shifts)
            @inbounds x .-= opt.quad_weights[i]*opt.krylov_solver.x[i]
        end
    end

    # println(opt.krylov_solver.stats.status)

    return
end
