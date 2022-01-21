#=
Author: Cooper Simpson

All functionality to interact with Flux.jl and support optimizing models using
newton type methods.
=#
using .Flux

#=
Optimizer for stochastic cubic Newton type methods.
=#
mutable struct StochasticCubicNewton
    optimizer::CubicNewtonOptimizer
    hessianSampleSize::UInt16
end

#=
Custom Flux training function for a StochasticCubicNewton optimizer

Input:
    model :: model + loss function
    _ :: model parameters (see below)
    data :: training data
    opt :: cubic newton optimizer

Because we extract the parameters from the model via a call to destructure, we
assume that all parameters there are being updated, and we don't need the
parameters passed in this function call.
=#
function Flux.Optimise.train!(model, _, data, opt::StochasticCubicNewton)
    #construct function compatible with optimizer
    params, re = Flux.destructure(model)
    f(θ, z) = re(θ)(z...)

    for batch in data
        #function of the parameters only
        f(θ) = θ -> f(θ, batch)

        #extract sub-sampled batch for hvp
        n = size(data, 1)
        subSample = selectdim(data, 1, rand(1:n, opt.hessianSampleSize))

        #compute gradients
        loss, back = pullback(f, params)
        grads = back(one.(loss))

        #make an update step
        opt.optimizer(f, params, grads, HVPOperator(θ -> f(θ, subSample), params), sum(loss))
    end
end
