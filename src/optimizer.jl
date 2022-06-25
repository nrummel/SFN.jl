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
NOTE: FGQ cant currently handle anything other than Float64
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
Computes an update step according to the shifted Lanczos-CG update rule.

Input:
    f :: scalar valued function
    x :: current iterate
    grads :: function gradients
    hess :: hessian operator
=#
function step!(opt::RSFNOptimizer, f::Function, x::T, grads::T, hess::HvpOperator) where T<:AbstractVector
    #compute regularization
    λ = norm(grads)^opt.p

    #compute shifts
    shifts = opt.nodes .+ λ

    #compute CG Lanczos quadrature integrand ((tᵢ²+λₖ)I+Hₖ²)⁻¹gₖ
    #NOTE: Could pre-allocate the space for the integrand
    integrand = solve!(opt.krylov_solver, hess, grads, shifts)

    #evaluate quadrature and update
    for (i, w) in enumerate(opt.quad_weights)
        @. x -= w*opt.solver.x[i]
    end

    return
end
