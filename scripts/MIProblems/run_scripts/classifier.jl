using DrWatson
@quickactivate

using master_thesis
using master_thesis: dist_knn
using Flux
using Distances

include(srcdir("point_cloud.jl"))
include(srcdir("mill_data.jl"))

r = parse(Float64, ARGS[1])
ratios = (r, 0.5-r, 0.5)
dataset = ARGS[2]
# dataset = "animals"
checkpath = datadir("experiments", "MIProblems", dataset, "classifier", "seed=1")
mkpath(checkpath)

# check that the combination does not yet exist
function check_params(checkpath, parameters, r)
    files = readdir(checkpath)
    n = savename(parameters)
    b_par = map(f -> occursin(n, f), files)
    b_r = map(f -> occursin("r=$r", f), files)
    b = b_par .* b_r
    # any(b) ? (return false) : (return true)
    any(b) ? false : true
end

# sample model parameters
function sample_params()
    hdim = sample([8,16,32,64])         # hidden dimension
    ldim = sample([2,4,8,16])           # latent dimension (last layer before softmax layer)
    batchsize = sample([64, 128, 256])
    agg = sample(["SegmentedMean", "SegmentedMax", "SegmentedMeanMax"])   # HMill aggregation function
    activation = sample(["relu", "swish", "tanh"])
    return (hdim = hdim, ldim = ldim, batchsize = batchsize, agg = agg, activation = activation)
end

# for small number of labels, kNN can fail, this does not fail
function try_catch_knn(DM, yk, y, k)
    try
        knn_v, kv = findmax(k -> dist_knn(k, DM, yk, y)[2], 1:k)
        return knn_v, kv
    catch e
        @warn "Caught error, reduced k: $k -> $(k-1)"
        try_catch_knn(DM, yk, y, k-1)
    end
end

function experiment(dataset::String, pvec, seed, ratios, max_train_time)
    @info "Starting loop for seed no. $seed."

    data, labels = load_multiclass(dataset)

    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data, labels; ratios=ratios, seed=seed)
    classes = sort(unique(yk))
    n = c = length(classes)
    Xval, yval, Xu, yu = validation_data(yk, Xu, yu, seed, classes)

    # and encode labels to onehot
    Xtrain = Xk
    ytrain = yk
    yoh_train = Flux.onehotbatch(ytrain, classes)
    activation = eval(Symbol(pvec.activation))
    aggregation = eval(Symbol(pvec.agg))

    # create a simple classificator model
    mill_model = reflectinmodel(
        Xtrain,
        d -> Dense(d, pvec.hdim, activation),
        aggregation
    )
    model = Chain(
            mill_model, Mill.data,
            Dense(pvec.hdim, pvec.hdim, activation), Dense(pvec.hdim, pvec.hdim, activation),
            Dense(pvec.hdim, pvec.ldim), Dense(pvec.ldim, n)
    )

    opt = ADAM()
    loss(x, y) = Flux.logitcrossentropy(model(x), y)
    accuracy(x, y) = round(mean(Flux.onecold(model(x), classes) .== y), digits=3)

    function minibatch()
        ix = sample(1:nobs(Xk), pvec.batchsize)
        xb = reindex(Xk, ix)
        yb = yoh_train[:, ix]
        xb, yb
    end

    best_model = deepcopy(model)
    best_acc = 0

    @info "Starting training with parameters $(pvec)..."
    start_time = time()

    while time() - start_time < max_train_time
        batches = map(_ -> minibatch(), 1:10)
        Flux.train!(loss, Flux.params(model), batches, opt)
        acc = accuracy(Xk, yk)
        # @show accuracy(Xk, yk)
        if acc >= best_acc
            # @show accuracy(Xk, yk)
            # @show accuracy(Xval, yval)
            # @show accuracy(Xt, yt)
            best_acc = acc
            best_model = deepcopy(model)
        end
    end
    @info "Training finished."

    ####################
    ### Save results ###
    ####################

    # results only for the best model
    predict_label(X) = Flux.onecold(best_model(X), classes)
    best_accuracy(x, y) = round(mean(Flux.onecold(best_model(x), classes) .== y), digits=5)

    # accuracy results
    train_acc = best_accuracy(Xk, yk)      # known labels
    val_acc = best_accuracy(Xval, yval)    # validation - used for hyperparameter choice
    test_acc = best_accuracy(Xt, yt)       # test data - this is the reference accuracy of model quality

    # confusion matrix on test data
    cm, df = confusion_matrix(classes, Xt, yt, predict_label)
    # @show df

    # kNN
    enc = best_model[1:end-1](Xk)
    enc_v = best_model[1:end-1](Xval)
    enc_u = best_model[1:end-1](Xu)
    enc_t = best_model[1:end-1](Xt)

    DMv = pairwise(Euclidean(), enc_v, enc)
    knn_v, kv = try_catch_knn(DMv, yk, yval, 10)

    # DMu = pairwise(Euclidean(), enc_u, enc)
    # knn_u, ku = findmax(k -> dist_knn(k, DMu, yk, yu)[2], 1:5)

    DMt = pairwise(Euclidean(), enc_t, enc)
    knn_t, kt = try_catch_knn(DMt, yk, yt, 10)

    results = Dict(
        :modelname => "classifier",
        :parameters => pvec,
        :train_acc => train_acc,
        :val_acc => val_acc,
        :test_acc => test_acc,
        :knn_v => knn_v,
        :kv => kv,
        #:knn_u => knn_u,
        #:ku => ku,
        :knn_t => knn_t,
        :kt => kt,
        :model => best_model,
        :CM => (cm, df),
        :seed => seed, 
        :r => r,
    )
    @info "Results for seed no. $seed calculated, saving..."

    nm = savename(savename(pvec), results, "bson")
    # @show nm
    safesave(datadir("experiments", "MIProblems", dataset, "classifier", "seed=$seed", nm), results)
    @info "Results saved."
end


max_train_time = 60*5

for k in 1:500
    parameters = sample_params()
    if check_params(checkpath, parameters, r)
        Threads.@threads for seed in 1:15
            experiment(dataset, parameters, seed, ratios, max_train_time)
        end
        break
    else
        @info "Parameters already present, trying new ones."
    end
end
