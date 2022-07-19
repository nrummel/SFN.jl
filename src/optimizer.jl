#=
Author: Cooper Simpson

R-SFN optimizer.
=#

using FastGaussQuadrature: gausslaguerre
using Krylov: CgLanczosShiftSolver, cg_lanczos_shift!

#=
R-SFN optimizer struct.
=#
mutable struct RSFNOptimizer{T1,T2<:Real, T3<:AbstractFloat, S<:AbstractVector{T3}}
    M::T1 #hessian lipschitz constant
    p::T2 #regularization power
    ϵ::T3#regularization minimum
    quad_nodes::S #quadrature nodes
    quad_weights::S #quadrature weights
    krylov_solver::CgLanczosShiftSolver #krylov inverse mat vec solver
end

#=
Outer constructor

NOTE: FGQ cant currently handle anything other than Float64

Input:
    dim :: dimension of parameters
    type :: parameter type
    M :: hessian lipschitz constant
    p :: regularization power
    ϵ :: regularization minimum
    quad_order :: number of quadrature nodes
=#
function RSFNOptimizer(dim::Int, type::Type{<:AbstractVector{T3}}=Vector{Float64}; M::T1=1, p::T2=1, ϵ::T3=eps(Float64), quad_order::Int=32) where {T1,T2<:Real, T3<:AbstractFloat}
    #krylov solver
    solver = CgLanczosShiftSolver(dim, dim, quad_order, type)

    #quadrature
    nodes, weights = gausslaguerre(quad_order, 0.0, reduced=true)

    @. weights = (2/pi)*weights*exp(nodes)
    @. nodes = nodes^2

    return RSFNOptimizer(M, p, ϵ, nodes, weights, solver)
end

#=
Repeatedly applies the R-SFN iteration to minimize the function.

Input:
    opt :: RSFNOptimizer
    x :: initialization
    f :: scalar valued function
    itmax :: maximum iterations
=#
function minimize!(opt::RSFNOptimizer, x::S, f::F; itmax::Int=1000) where {S<:AbstractVector{<:AbstractFloat}, F}
    logger = Logger()

    grads = similar(x)
    Hop = RHvpOperator(f, x)

    for i = 1:itmax
        #construct gradient and hvp operator
        loss, back = pullback(f, x)
        grads .= back(one(loss))[1]
        logger.gcalls += 1

        #iterate
        step!(opt, x, f, grads, Hop)

        logger.hcalls += Hop.nProd

        update!(Hop, x)
    end

    return logger
end

#=
Computes an update step according to the shifted Lanczos-CG update rule.

Input:
    opt :: RSFNOptimizer
    x :: current iterate
    f :: scalar valued function
    grads :: function gradients
    hess :: hessian operator
=#
function step!(opt::RSFNOptimizer, x::S, f::F, grads::S, Hop::HvpOperator) where {S<:AbstractVector{<:AbstractFloat}, F}
    #compute regularization
    λ = (opt.M*norm(grads))^opt.p + opt.ϵ

    #compute shifts
    shifts = opt.quad_nodes .+ λ

    #compute CG Lanczos quadrature integrand ((tᵢ²+λₖ)I+Hₖ²)⁻¹gₖ
    cg_lanczos_shift!(opt.krylov_solver, Hop, grads, shifts)

    #evaluate quadrature and update
    @simd for i = 1:size(shifts, 1)
        @inbounds x .-= opt.quad_weights[i].*opt.krylov_solver.x[i]
    end

    return nothing
end
