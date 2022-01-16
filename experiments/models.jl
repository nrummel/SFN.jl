#=
Build a simple dense NN with paramaters ≈ data points * over paramaterization factor.

Input:
    input_dim :: dimension of input
    output_dim :: dimension of output
    num_data :: number of training data points
    op_factor :: over paramaterization factor
    num_layers :: depth of network (optional)
=#
function build_dense(input_dim, output_dim, num_data, op_factor, num_layers=3)
    params = num_data*op_factor #approx number of paramaters
    hidden = ceil(params - params*(input_dim + output_dim))/num_layers) #hidden layer params

    model = Flux.Dense(input_dim, hidden, relu)
    for i = 2:num_layers-1
        model = Flux.Chain(model, Flux.Dense(hidden, hidden, relu))
    end
    model = Flux.Chain(model, Flux.Dense(hidden, output_dim))

    return model
end

#=
Return model prediction

Input:
    model :: NN model
    x :: input data
=#
predict = (model, x) -> return Flux.softmax(model(x))

#=
Compute model accuracy on given data

Input:
    model :: NN model
    dataloader :: data loader
    labels :: data labels
=#
function accuracy(model, dataloader, labels)
    correct, total = 0, 0
    for x, y in dataloader
        ̂y = Flux.onehotbatch(predict(model, x), labels)

        total += size(labels, 1)
        correct += sum(̂y == y)

    return correct/total
end
