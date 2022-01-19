#=
Author: Cooper Simpson

Main experiment logic.
=#

using CubicNewton
using Flux: Descent

include("models.jl")
include("mnist.jl")

#=
Setup hyperparameters
=#
batchSize = 32
epochs = 1

#=
Build and train
=#

#build model
# model = build_dense(28*28, 10, 6e4, 1)
model = mnist_dense()

#load data
train, test = mnist(batchSize)

#select optimizer
# opt = Descent()
opt = ShiftedLanczosCG()

#add loss to model
loss(x, y) = logitcrossentropy(y, model(x))

#train
for epoch = 1:epochs
    println("Epoch: $epochs")
    Flux.train!(loss, params(model), train, opt)
end

println("Accuracy: $(accuracy(model, test, 0:9))")
