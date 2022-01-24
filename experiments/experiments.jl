#=
Author: Cooper Simpson

Main experiment logic.
=#

using Flux
using CUDA
using CubicNewton: StochasticCubicNewton

include("models.jl")
include("datasets.jl")

#=
Setup hyperparameters
=#
batchSize = 32
epochs = 1
order = 1

#=
Build and train
=#

#build model
model = build_dense(28*28, 10, 1e4, 1) |> gpu

#load data
trainLoader, testLoader = mnist(batchSize)

#select optimizer and add loss function
if order == 1
    opt = Flux.Descent()
    ps = params(model)
    loss(x,y) = Flux.Losses.logitcrossentropy(model(x), y)
elseif order == 2
    opt = StochasticCubicNewton()
    ps, re = Flux.destructure(model)
    loss(θ,x,y) = Flux.Losses.logitcrossentropy(re(θ)(x), y)
end

#check accuracy before hand
println("Accuracy: $(accuracy(model, testLoader, 0:9))")

#train
for epoch = 1:epochs
    println("Epoch: $epoch")
    Flux.Optimise.train!(loss, ps, trainLoader, opt)
end

#Need to do this because ps is a copy of the parameters
if order == 2
    model = re(ps)
end

println("Accuracy: $(accuracy(model, testLoader, 0:9))")
