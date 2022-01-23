#=
Author: Cooper Simpson

Functionality for working with datasets.
=#

using MLDatasets: MNIST
using Flux.Data: DataLoader

#=
Load the MNIST train and testing datasets

Input:
    batchSize :: batch size (optional)
=#
function mnist(batchSize=32)
    #setup data
    dir = "./data/MNIST"
    train = MNIST.traindata(Float32, dir=dir)
    test = MNIST.testdata(Float32, dir=dir)

    trainLoader = DataLoader((train[1], Flux.onehotbatch(train[2], 0:9)), batchsize=batchSize, shuffle=true)
    testLoader = DataLoader(test, batchsize=batchSize, shuffle=true)

    return trainLoader, testLoader
end
