#=
Author: Cooper Simpson

Main experiment logic.
=#

using Flux
using CubicNewton: StochasticCubicNewton

include("models.jl")
include("datasets.jl")

#=
Setup hyperparameters
=#
batchSize = 32
epochs = 1
order = 2

#=
Build and train
=#

#build model
model = build_dense(28*28, 10, 1e4, 1)

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

#train
for epoch = 1:epochs
    println("Epoch: $epochs")
    Flux.Optimise.train!(loss, ps, trainLoader, opt)
end

println("Accuracy: $(accuracy(model, testLoader, 0:9))")
