#=
Author: Cooper Simpson

Functionality for working with datasets.
=#

using .Flux
using DataLoaders
using MLDatasets: MNIST

#=
Custom collate function for Flux onehot vectors, because the standard implementations don't work.
=#
DataLoaders.collate(samples::AbstractVector{<:Flux.OneHotArray}) = hcat(samples...) |> gpu

#=
MNIST dataset
=#
struct MNISTDataset
    dir::String
    nSamples::Int
    train::Bool
end
TrainMNIST(dir::String) = MNISTDataset(dir, 60000, 1)
TestMNIST(dir::String) = MNISTDataset(dir, 10000, 0)

DataLoaders.LearnBase.nobs(dataset::MNISTDataset) = dataset.nSamples

function DataLoaders.LearnBase.getobs(dataset::MNISTDataset, i::Int)
    if dataset.train
        (data, label) = MNIST.traindata(Float32, i, dir=dataset.dir)
        label = Flux.onehotbatch([label], 0:9)
    else
        (data, label) = MNIST.testdata(Float32, i, dir=dataset.dir)
    end

    return (data |> gpu, label)
end

function DataLoaders.LearnBase.getobs!(buffer, dataset::MNISTDataset, i::Int)
    if dataset.train
        (data, label) = MNIST.traindata(Float32, i, dir=dataset.dir)
        buffer[2] .= Flux.onehotbatch([label], 0:9)
    else
        (data, label) = MNIST.testdata(Float32, i, dir=dataset.dir)
        buffer[2] .= label
    end
    buffer[1] .= data |> gpu
end

function mnist_lazy(batchSize::Int, dir::String)
    train, test = TrainMNIST(dir), TestMNIST(dir)

    trainLoader = DataLoaders.DataLoader(train, batchSize, collate=true, buffered=false)
    testLoader = DataLoaders.DataLoader(test, batchSize, collate=true, buffered=false)

    return trainLoader, testLoader
end

#=
Load all data onto GPU
=#
function mnist(batchSize::Int, dir::String)
    train = MNIST.traindata(Float32, dir=dir)
    test = MNIST.testdata(Float32, dir=dir)

    trainLoader = Flux.DataLoader((train[1] |> gpu, Flux.onehotbatch(train[2], 0:9) |> gpu), batchsize=batchSize, shuffle=true)
    testLoader = Flux.DataLoader((test[1] |> gpu, test[2] |> gpu), batchsize=batchSize, shuffle=false)

    return trainLoader, testLoader
end

#=
CUDA sparse vector stuff for making onehot vectors and arrays
=#
# DataLoaders.collate(samples::AbstractVector{<:CUDA.CUSPARSE.CuSparseVector{T, N}}) where {T, N} = hcat(samples...)
# using SparseArrays: sparsevec, sparse
# sparsevec([label+1], [true], 10)
# sparse(train[2] .+= 1, 1:size(train[2],1), [true for i in 1:size(train[2],1)])
