using MLDatasets: MNIST

function mnist(model, opt, epochs=1, batchsize=32)
    #setup data
    dir = "./data/MNIST"
    train = MNIST.traindata(dir=dir)
    test = MNIST.testdata(dir=dir)

    train = Flux.DataLoader((train[1], Flux.onehotbatch(train[2], 0:9)), batchsize=batchsize, shuffle=true)
    test = Flux.DataLoader((test[1], Flux.onehotbatch(test[2], 0:9)), batchsize=batchsize, shuffle=true)

    #loss function, z = (x, y)
    loss(z) = Flux.logitcrossentropy(z[2], model(z[1]))

    #train
    for epoch = 1:epochs
        Flux.train!(loss, train, opt)
    end

    println("Accuracy: ", accuracy(model, test, 0:9))
end
