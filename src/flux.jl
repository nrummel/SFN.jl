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
        # n = size(X, ndims(X))
        #
        # indices = rand(1:n, ceil(Int, opt.hessianSampleFactor*n))
        # subX = selectdim(X, ndims(X), indices)
        # #NOTE: Ideally selectdim would handle this properly, so we wouldn't
        # #have to convert indices into cuda array
        # subY = Base.unsafe_view(Y, Base.Slice(Base.OneTo(size(Y,1))), subX.indices[ndims(X)])
        subX, subY = X, Y

        Hop = HvpOperator(θ -> f(θ, subX, subY), ps)

        #compute gradients
        loss, back = pullback(θ -> f(θ, X, Y), ps)
        grads = back(one(loss))[1]

        #make an update step
        print("...")
        step!(opt.optimizer, θ -> f(θ, X, Y), ps, grads, Hop, loss)
        println("step")
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
