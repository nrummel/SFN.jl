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
    optimizer::CubicNewtonOptimizer
    hessianSampleFactor::Float32
    log = Logger()
end

#=
Constructor.

Input:
    type :: type of paramaters, gradients, etc.
    dim :: number of parameters (length of gradient)
    hessianSampleFactor :: hessian sub sample factor in (0,1] (optional)
=#
function StochasticCubicNewton(type::Type, dim::Int, hessianSampleFactor=0.1)
    if !((0.0 < hessianSampleFactor) && (hessianSampleFactor <= 1.0))
        throw(ArgumentError("Hessian sample factor not in range (0,1]."))
    end

    return StochasticCubicNewton(optimizer=ShiftedLanczosCG(type, dim), hessianSampleFactor=hessianSampleFactor)
end

#=
Custom Flux training function for a StochasticCubicNewton optimizer

Input:
    f :: model + loss function
    ps :: model params
    trainLoader :: training data
    opt :: cubic newton optimizer
=#
function Flux.Optimise.train!(f::Function, ps::T, trainLoader, opt::StochasticCubicNewton) where T<:AbstractVector
    for (X, Y) in trainLoader
        #build hvp operator using subsampled batch
        n = size(X, ndims(X))

        idx = rand(1:n, ceil(Int, opt.hessianSampleFactor*n))
        # subX = selectdim(X, ndims(X), idx)
        # subY = selectdim(Y, ndims(Y), idx)
        subX = X[Tuple([Colon() for i in 1:ndims(X)-1])..., idx]
        subY = Y[:, idx]

        # update!(opt.Hop, θ -> f(θ, subX, subY), ps)
        Hop = HvpOperator(θ -> f(θ, subX, subY), ps)

        #compute gradients
        loss, back = pullback(θ -> f(θ, X, Y), ps)
        grads = back(one(loss))[1]

        #make an update step
        step!(opt.optimizer, θ -> f(θ, X, Y), ps, grads, Hop, loss)

        opt.log.hvps += Hop.nProd
    end
end

#=
Flatten a models parameters into a single vector, and then create a new model
that references these flattened parameters.

NOTE: Assumes everything is trainable

Input:
    model :: Flux model
=#
function make_flat(model)
    #grab all the paramaters
    ps = AbstractVector[]
    fmap(model) do p
        p isa AbstractArray && push!(arrays, vec(p))
        return x
    end
    flat_ps = reduce(vcat, ps)

    #Make a new model with views into flattened parameters
    offset = Ref(0)
    out = fmap(model) do p
        p isa AbstractArray || return p
        y = view(flat_ps, offset[] .+ (1:length(p)))
        offset[] += length(p)
        return reshape(y, size(p))
    end

    #return flattened parameters and new model
    return flat, out
end
