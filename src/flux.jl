#=
Author: Cooper Simpson

All functionality to interact with Flux.jl and support optimizing models using
newton type methods.
=#
using .Flux

#=
Optimizer for stochastic cubic Newton type methods.
=#
Base.@kwdef mutable struct StochasticCubicNewton
    optimizer::CubicNewtonOptimizer = ShiftedLanczosCG()
    hessianSampleFactor = 0.1
end

#=
Custom Flux training function for a StochasticCubicNewton optimizer

Input:
    f :: model + loss function
    ps :: model params
    trainLoader :: training data
    opt :: cubic newton optimizer
=#
function Flux.Optimise.train!(f, ps, trainLoader, opt::StochasticCubicNewton)
    for (X, Y) in trainLoader
        #build hvp operator using subsampled batch
        n = size(X, ndims(X))

        indices = rand(1:n, ceil(Int, opt.hessianSampleFactor*n))
        subSampleX = selectdim(X, ndims(X), indices)
        subSampleY = selectdim(Y, ndims(Y), indices)

        Hop = HvpOperator(θ -> f(θ, subSampleX, subSampleY), ps)

        #compute gradients
        loss, back = pullback(θ -> f(θ, X, Y), ps)
        grads = back(one(loss))[1]

        #make an update step
        step!(opt.optimizer, θ -> f(θ, X, Y), ps, grads, Hop, loss)
    end
end
