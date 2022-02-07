#=
Author: Cooper Simpson

Main experiment logic.
=#

using Flux
using CUDA
using CubicNewton: StochasticCubicNewton
using Serialization

include("models.jl")
include("datasets.jl")

if Flux.use_cuda[] == false
    ErrorException("Not using GPU.")
end

CUDA.allowscalar(false)

#=
Setup hyperparameters
=#
batchSize = 256
epochs = 1
order = 1

#=
Build and train
=#

#build model
model = build_dense(28*28, 10, 6e4, 1) |> gpu

#load data
trainLoader, testLoader = mnist(batchSize)

#select optimizer and add loss function
if order == 1
    ps = params(model)
    loss(x,y) = Flux.Losses.logitcrossentropy(model(x), y)
    opt = Flux.Descent()
elseif order == 2
    ps, re = Flux.destructure(model)
    loss(θ,x,y) = _logitcrossentropy(re(θ)(x), y)
    opt = StochasticCubicNewton(typeof(ps), size(ps, 1))
end

#check GPU usage
println(CUDA.memory_status())

#check accuracy before hand
# println("Accuracy: $(accuracy(model, testLoader, 0:9))")
acc = Vector{Float32}(undef, epochs)

#train
for epoch = 1:epochs
    println("Epoch: $epoch")
    Flux.Optimise.train!(loss, ps, trainLoader, opt)

    #Need to do this because ps is a copy of the parameters
    if order == 2
        acc[epoch] = accuracy(re(ps), testLoader, 0:9)
    else
        acc[epoch] = accuracy(model, testLoader, 0:9)
    end
end

println("Final Accuracy: $(acc[epochs])")

# #Serialize logger
# if order == 2
#     serialize("results/order-$order", (acc, opt.log.hvps))
# else
#     serialize("results/order-$order", (acc,))
# end
