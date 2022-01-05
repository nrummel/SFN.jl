#=
Author: Cooper Simpson

Flux.jl support
=#
using .Flux

#=
Custom Flux training function for a CubicNewton optimizer

Input:
    model -> model + loss function
    data -> training data
    opt -> cubic newton optimizer
=#
function train!(model, data, opt::CubicNewtonOpt)
    #construct function compatible with optimizer
    params, re = Flux.destructure(model)
    f(θ, x) = re(θ)(x)

    for batch in data
        opt(f, params, batch)
    end

    return true
end

#=
Because we extract the parameters from the model via a call to destructure, we
assume that all parameters there are begin updated, and we don't need the
parameters passed in this function call.
=#
function Flux.Optimise.train!(model, params, data, opt::CubicNewtonOpt)
    train!(model, data, opt)
end
