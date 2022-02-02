#=
Author: Cooper Simpson

Main experiment logic.
=#

using Flux
using CUDA
using CubicNewton: StochasticCubicNewton

include("models.jl")
include("datasets.jl")

if Flux.use_cuda[] == false
    ErrorException("Not using GPU.")
end

CUDA.allowscalar(false)

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
model = build_dense(28*28, 10, 1e3, 1) |> gpu

#load data
trainLoader, testLoader = mnist(batchSize)

#select optimizer and add loss function
if order == 1
    ps = params(model)
    loss(x,y) = _logitcrossentropy(model(x), y)
    opt = Flux.Descent()
elseif order == 2
    ps, re = Flux.destructure(model)
    loss(θ,x,y) = _logitcrossentropy(re(θ)(x), y)
    opt = StochasticCubicNewton(typeof(ps), size(ps, 1))
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
