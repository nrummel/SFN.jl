#=
Author: Cooper Simpson

Functionality for building and using dense NNs.
=#

using Roots: find_zero

#=
Build a simple dense NN with paramaters ≈ data points * over paramaterization factor.

Input:
    inputDim :: dimension of input
    outputDim :: dimension of output
    numData :: number of training data points
    opFactor :: over paramaterization factor
    numLayers :: depth of network (optional)
    scale :: per-layer scale in parameters
=#
function build_dense(inputDim, outputDim, numData, opFactor, numLayers=3, scale=0.5)
    #calculate dimensions so that each layer has approximately equal parameters
    totalParams = numData*opFactor

    l = numLayers-2
    p(x, i, k) = x*scale^(i-1+k)
    params(x) = begin
        res = x*inputDim
        for i=1:l
            dᵢ, dₒ = p(x, i, 0), p(x, i, 1)
            res += dₒ*(dᵢ + 1)
        end
        return res += outputDim*(p(x, l, 1) + 1)
    end
    x = ceil(Int, find_zero(x -> params(x)-totalParams, (0, totalParams/inputDim)))

    model = Flux.Dense(inputDim, ceil(Int, p(x, 1, 0)), relu)∘Flux.flatten
    for i=1:l
        model = Flux.Dense(ceil(Int, p(x, i, 0)), ceil(Int, p(x, i, 1)), relu)∘model
    end
    return Flux.Dense(ceil(Int, p(x, l, 1)), outputDim)∘model
end

#=
Return model prediction.

Input:
    model :: NN model
    x :: input data
=#
predict(model, x) = Flux.softmax(model(x))

#=
Compute model accuracy on given data.

Input:
    model :: NN model
    dataLoader :: data loader
    labels :: data labels
=#
function accuracy(model, dataLoader, labels)
    correct, total = 0, 0
    for (x, y) in dataLoader
        pred = Flux.onecold(predict(model, x), labels)

        total += size(labels, 1)
        correct += sum(pred .== y)
    end

    return correct/total
end
