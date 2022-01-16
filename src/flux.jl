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
    hessian_sample_size::UInt16
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
function Flux.Optimise.train!(model, data, opt::StochasticCubicNewton)
    #construct function compatible with optimizer
    params, re = Flux.destructure(model)
    f(θ, x) = re(θ)(x)

    for batch in data
        #function of the parameters only
        f(θ) = θ -> f(θ, batch)

        #compute gradients
        loss, grads = withgradient(f, params)

        #extract sub-sampled batch for hvp
        n = size(data, 1)
        subsample = selectdim(data, 1, rand(1:n, opt.hessian_sample_size))

        #make an update step
        opt.optimizer(f, params, grads, HVPOperator(θ -> f(θ, subsample), params))
    end
end
