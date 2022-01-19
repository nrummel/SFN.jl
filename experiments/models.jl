#=
Author: Cooper Simpson

Functionality for building and using dense NNs.
=#

using Flux: flatten, Dense, softmax, onecold, logitcrossentropy, params

#=
Build a simple dense NN with paramaters ≈ data points * over paramaterization factor.

Input:
    inputDim :: dimension of input
    outputDim :: dimension of output
    numData :: number of training data points
    opFactor :: over paramaterization factor
    numLayers :: depth of network (optional)
=#
# function build_dense(inputDim, outputDim, numData, opFactor, numLayers=3)
#     #calculate dimensions so that each layer has approximately equal parameters
#     params = ceil(Int, numData*opFactor/numLayers)
#
#     return model
# end

function mnist_dense()
    return Flux.Chain(flatten, Dense(28*28, 72, relu),
                        Dense(72, 48, relu), Dense(48, 10))
end

#=
Return model prediction.

Input:
    model :: NN model
    x :: input data
=#
predict(model, x) = softmax(model(x))

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
        pred = onecold(predict(model, x), labels)

        total += size(labels, 1)
        correct += sum(pred .== y)
    end

    return correct/total
end
