using NearestNeighbors


# enc = embedding(Xk)
# tree = BruteTree(enc)

# enc_unknown = embedding(Xu)
# idxs, dists = knn(tree, enc_unknown, 10)

# b = map(idx -> length(unique(yk[idx])) == 1, idxs) |> BitVector
# ixs = (1:nobs(Xu))[b]
# accuracy(reindex(Xu, ixs), yu[ixs])

function classify_unknown(model, Xk, yk, Xu, yu, k, classes)
    embedding(X) = Mill.data(model[1](X))

    enc = embedding(Xk)
    tree = BruteTree(enc)

    enc_unknown = embedding(Xu)
    idxs, dists = knn(tree, enc_unknown, k)

    b = map(idx -> length(unique(yk[idx])) == 1, idxs) |> BitVector
    ixs = (1:nobs(Xu))[b]
    ixs_left = setdiff(1:nobs(Xu), ixs)
    
    Xknew = reindex(Xu, ixs)
    yknew = map(idx -> unique(yk[idx])[1], idxs[ixs])
    
    @info "Accuracy of inferred labels: $(mean(yu[ixs] .== yknew))."

    Xunew = reindex(Xu, ixs_left)
    yunew = yu[ixs_left]

    return Xknew, yknew, Xunew, yunew, ixs
end

function uniform_minibatch()
    k = round(Int, batchsize / length(classes))

    ixs = []
    for c in classes
        chosen = findall(x -> x == c, yk)
        ix = sample(chosen, k)
        push!(ixs, ix)
    end

    ix = vcat(ixs...)
    xb = reindex(Xk, ix)
    yb = yoh_train[:, ix]
    xb, yb
end

function train_classifier(hdim, batchsize, activation, aggregation, Xk, yk, Xval, yval, Xu, yu; k, max_train_time = 60*5)
    # prepare data
    classes = sort(unique(yk))
    Xtrain = Xk
    ytrain = yk
    yoh_train = Flux.onehotbatch(ytrain, classes)

    function minibatch()
        ix = sample(1:nobs(Xk), batchsize)
        xb = reindex(Xk, ix)
        yb = yoh_train[:, ix]
        xb, yb
    end
    function uniform_minibatch()
        k = round(Int, batchsize / length(classes))
    
        ixs = []
        for c in classes
            chosen = findall(x -> x == c, yk)
            ix = sample(chosen, k)
            push!(ixs, ix)
        end
    
        ix = vcat(ixs...)
        xb = reindex(Xk, ix)
        yb = yoh_train[:, ix]
        xb, yb
    end

    # initialize the classifier
    mill_model = reflectinmodel(
        Xtrain,
        d -> Dense(d, hdim, activation),
        SegmentedMeanMax
    )
    model = Chain(
            mill_model, Mill.data,
            Dense(hdim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, length(classes))
    )

    # create loss and accuracy functions
    loss(x, y) = Flux.logitcrossentropy(model(x), y)
    accuracy(x, y) = round(mean(classes[Flux.onecold(model(x))] .== y), digits=4)
    predict_label(X) = Flux.onecold(model(X), classes)
    opt = ADAM()

    @info "Starting training..."    
    yval_oh = Flux.onehotbatch(yval, classes)
    best_val_acc = 0
    best_model = deepcopy(model)

    start_time = time()
    while time() - start_time < max_train_time

        batches = map(_ -> uniform_minibatch(), 1:10)
        Flux.train!(loss, Flux.params(model), batches, opt)

        val_acc = accuracy(Xval, yval)
        if val_acc >= best_val_acc
            @info "\nValidation accuracy = $val_acc." 
            @show accuracy(Xk, yk)

            best_val_acc = val_acc
            best_model = deepcopy(model)
        else
            print(".")
        end
    end
    @info "Training finished."

    Xknew, yknew, Xunew, yunew = classify_unknown(model, Xk, yk, Xu, yu, k, classes)

    return cat(Xk, Xknew), vcat(yk, yknew), Xunew, yunew, best_model, best_val_acc
end