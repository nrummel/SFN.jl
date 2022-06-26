#=
Author: Cooper Simpson

R-SFN optimizer.
=#

using FastGaussQuadrature: gausslaguerre
using Krylov: CgLanczosShiftSolver, solve!

#=
R-SFN optimizer struct.
=#
mutable struct RSFNOptimizer{T<:AbstractFloat, S<:AbstractVector{T}}
    krylov_solver::CgLanczosShiftSolver #krylov inverse mat vec solver
    quad_nodes::S #quadrature nodes
    quad_weights::S #quadrature weights
    p::T #regularization power
    ϵ::T #regularization minimum
end

#=
Outer constructor

NOTE: FGQ cant currently handle anything other than Float64

Input:
    dim :: dimension of parameters
    quad_order :: number of quadrature nodes
    p :: regularization power
    ϵ :: regularization minimum
=#
function RSFNOptimizer(dim::Int, quad_order::Int=32, S::Type{<:AbstractVector{T}}=Vector{Float64}, p::T=1.0, ϵ::T=eps(T)) where T<:AbstractFloat
    #krylov solver
    solver = CgLanczosShiftSolver(dim, dim, quad_order, T)

    #quadrature
    nodes, weights = gausslaguerre(quad_order)
    @. nodes = nodes^2 #will always need nodes squared

    return RSFNOptimizer{S, T}(solver, nodes, weights, p, ϵ)
end

#=
Repeatedly applies the R-SFN iteration to minimize the function.

Input:
    opt :: RSFNOptimizer
    x :: initialization
    f :: scalar valued function
    itmax ::
=#
function minimize!(opt::RSFNOptimizer, x::T, f::Function, itmax::Int=1e3) where T<:AbstractVector
    grads = similar(x)

    for i = 1:itmax
        #construct gradient and hvp operator
        loss, back = pullback(f, x)
        grads .= back(one(loss))[1]

        Hop = HvoOperator(f, x)

        #iterate
        step!(opt, x, f, grads, Hop)
    end
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
function step!(opt::RSFNOptimizer, x::T, f::Function, grads::T, Hop::HvpOperator) where T<:AbstractVector
    #compute regularization
    λ = norm(grads)^opt.p

    #compute shifts
    shifts = opt.nodes .+ λ

    #compute CG Lanczos quadrature integrand ((tᵢ²+λₖ)I+Hₖ²)⁻¹gₖ
    solve!(opt.krylov_solver, Hop, grads, shifts)

    #evaluate quadrature and update
    for (i, w) in enumerate(opt.quad_weights)
        @. x -= w*opt.solver.x[i]
    end

    return
end
