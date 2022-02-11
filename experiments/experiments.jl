
#=
Author: Cooper Simpson

Main experiment logic.
=#

using Flux
using CUDA
using Serialization
import CubicNewton as CN

#=
Check on GPU
=#
if Flux.use_cuda[] == false
    ErrorException("Not using GPU.")
end

CUDA.allowscalar(false)

#=
Setup test
=#
batchSize = 256
epochs = 1
order = 2

#=
Build and train
=#

#build model
model = CN.build_dense(28*28, 10, 6e4, 1) |> gpu

#load data
trainLoader, testLoader = CN.mnist(batchSize, "./data/MNIST")

#select optimizer and add loss function
if order == 1
    ps = params(model)
    loss(x,y) = Flux.Losses.logitcrossentropy(model(x), y)
    opt = Flux.Descent()
elseif order == 2
    ps, re = Flux.destructure(model)
    loss(θ,x,y) = CN._logitcrossentropy(re(θ)(x), y)
    opt = CN.StochasticCubicNewton(typeof(ps), size(ps, 1))
end

#check GPU usage
#TODO: Log this information somewhere instead
println(CUDA.memory_status(), "\n")

#testing accuracy
acc = Vector{Float32}(undef, epochs)

#train
for epoch = 1:epochs
    println("Epoch: $epoch")
    Flux.Optimise.train!(loss, ps, trainLoader, opt)

    #Need to do this because ps is a copy of the parameters
    if order == 2
        acc[epoch] = CN.accuracy(re(ps), testLoader, 0:9)
    else
        acc[epoch] = CN.accuracy(model, testLoader, 0:9)
    end
end

println("Training completed, saving data...")

# #Serialize logger
# if order == 2
#     serialize("results/order-$order", (acc, opt.log.hvps))
# else
#     serialize("results/order-$order", (acc,))
# end
