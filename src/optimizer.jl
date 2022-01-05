#=
Author: Cooper Simpson

Main cubic newton optimization functionality.
=#

include("sub_problem/sub_problem.jl")

#=
Cubic Newton optimizer definition
=#
mutable struct CubicNewtonOpt
    σ::Real
    η₁::Real
    η₂::Real
    γ₁::Real
    γ₂::Real
end
CubicNewtonOpt(σ=1, η₁=0.1, η₂=0.9, γ₁=2, γ₂=2) = CubicNewtonOpt(σ, η₁, η₂, γ₁, γ₂)

#=
Main cubic newton loop.

Input:
    f(θ, x) -> function returning a scalar (e.g. model + loss function)
    params -> parameters of f (e.g. model weights)
    data -> inputs to f (e.g. minibatch)
    opt -> cubic newton optimizer
=#
function (opt::CubicNewtonOpt)(f, params, data)
    #compute gradients
    loss, grads = withgradient(θ -> f(θ, data), params)

    #extract sub-sampled batch for hvp
    n = size(data, 1)
    subsample = selectdim(data, 1, rand(1:n, ceil(n*.1)))

    #solve sub-problem to yield descent direction s and minimum m
    s, m = _cubic_minimize(f, params, subsample, grads[1], opt.σ)

    #evaluate descent direction, adapt hyperparameters, update parameters
    params .-= s
    ρ = (f(params, data) - loss)/m

    #bad update
    if ρ<opt.η₁
        opt.σ *= opt.γ₁
    #great update
    elseif ρ>=opt.η₂
        opt.σ = max(opt.σ/opt.γ₂, 1e-16)
    end

    return true
end
