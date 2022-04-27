function classify_unknown(model, Xk, yk, Xu, yu, threshold, classes)
    probs = softmax(model(Xu))
    b = map(x -> any(x .> threshold), eachcol(probs))
    ixs = (1:nobs(Xu))[b]
    ixs_left = setdiff(1:nobs(Xu), ixs)
    
    Xknew = reindex(Xu, ixs)
    yknew = Flux.onecold(model(Xknew), classes)
    
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

function train_classifier(hdim, batchsize, activation, aggregation, thr, Xk, yk, Xval, yval, Xu, yu; k, max_train_time = 60*5)
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
    patience = 0
    max_patience = 1000

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
            patience = 0
        else
            print(".")
            patience += 1
            if patience > max_patience
                @info "Patience exceeded, training stopped."
                break
            end
        end
    end
    @info "Training finished."

    Xknew, yknew, Xunew, yunew = classify_unknown(best_model, Xk, yk, Xu, yu, thr, classes)
    # Xknew, yknew, Xunew, yunew = classify_unknown_knn(model, Xk, yk, Xu, yu, k, classes)

    return cat(Xk, Xknew), vcat(yk, yknew), Xunew, yunew, best_model, best_val_acc
end

using NearestNeighbors
function classify_unknown_knn(model, Xk::T, yk, Xu::T, yu, k::Int, classes; quant=0.2) where T <: AbstractMillNode
    # calculate encodings
    enc = model(Xk)
    enc_unknown = model(Xu)

    # create tree and find k-nearest neighbors for each point in Xu
    tree = BruteTree(enc)
    idxs, dists = knn(tree, enc_unknown, k)

    # calculate quantile mean distance
    md = mean.(dists)
    q = quantile(md, quant)

    # filter only the samples which have k train neighbors from same class
    # and the distance is in the 20% quantile
    bk = map(idx -> length(unique(yk[idx])) == 1, idxs) |> BitVector
    bq = md .< q
    b = bk .* bq
    ixs = (1:nobs(Xu))[b]
    ixs_left = setdiff(1:nobs(Xu), ixs)
    
    Xknew = reindex(Xu, ixs)
    yknew = map(idx -> unique(yk[idx])[1], idxs[ixs])
    
    @info "Accuracy of inferred labels: $(mean(yu[ixs] .== yknew))."

    Xunew = reindex(Xu, ixs_left)
    yunew = yu[ixs_left]

    return Xknew, yknew, Xunew, yunew, ixs
end

function train_knn(Xk, yk, Xval, yval, Xu, yu, parameters; max_train_time = 60*5)
    # prepare data
    classes = sort(unique(yk))
    Xtrain = Xk
    ytrain = yk
    yoh_train = Flux.onehotbatch(ytrain, classes)
    c = length(classes)

    batchsize=64
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
        y = ytrain[ix]
        xb, yb, y
    end

    # model
    feature_model, W = arcface_constructor(Xtrain, c; parameters...)
    opt = ADAM()
    ps = Flux.params(W, feature_model)

    # parameters
    s = 64f0
    m = 0.5f0

    # create loss and accuracy functions
    ls = eval(Symbol(parameters.loss))
    loss(x, y, yl) = ls(feature_model, W, m, s, x, y, yl)
    # loss(x, y, yk) = lossf(x, y, yk)
    yoh_val = Flux.onehotbatch(yval, classes)
    best_val_loss = Inf
    best_model = deepcopy([feature_model, W])
    patience = 0
    max_patience = 2000

    @info "Starting training..."
    start_time = time()
    while time() - start_time < max_train_time

        batches = map(_ -> uniform_minibatch(), 1:10)
        Flux.train!(loss, ps, batches, opt)

        val_loss = loss(Xval, yoh_val, yval)
        if val_loss <= best_val_loss
            @info "Validation loss = $(round(val_loss, digits=5))."

            best_val_loss = val_loss
            best_model = deepcopy([feature_model, W])
            patience = 0
        else
            print(".")
            patience += 1
            if patience > max_patience
                @info "Patience exceeded, training stopped."
                break
            end
        end
    end
    @info "Training finished."

    Xknew, yknew, Xunew, yunew = classify_unknown_knn(feature_model, Xk, yk, Xu, yu, parameters.k, classes; quant=parameters.quant)

    return cat(Xk, Xknew), vcat(yk, yknew), Xunew, yunew, best_model, best_val_loss
end

function predict_labels(model, k, Xk, yk, X, y)
    enc, enc_test = model(Xk), model(X)
    DM = pairwise(Euclidean(), enc_test, enc)
    ypred, acc = dist_knn(k, DM, yk, y)
end
accuracy(X, y) = predict_labels(model, parameters.k, _Xk, _yk, X, y)[2]

# map(model -> predict_labels(model[1], parameters.k, _Xk, _yk, Xval, yval), best_models)