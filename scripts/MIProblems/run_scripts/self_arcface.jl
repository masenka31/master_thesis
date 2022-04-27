using DrWatson
@quickactivate

using Flux, Mill
using ClusterLosses

using master_thesis
using master_thesis: encode, arcface_constructor

using Statistics
using Distances, Clustering

# prerequisities
include(srcdir("point_cloud.jl"))
include(srcdir("mill_data.jl"))
include(scriptsdir("MNIST", "semisup.jl"))

# args
r = parse(Float64, ARGS[1])     # controls ratio of known labels
ratios = (r, 0.5-r, 0.5)        # get the ratios
dataset = ARGS[2]
checkpath = datadir("experiments", "MIProblems", dataset, "self_arcface", "seed=1")
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

function sample_params()
    hdim = sample([8,16,32,64])
    odim = sample([2,4,8,16])
    while odim > hdim
        odim = sample([2,4,8,16])
    end
    activation = sample(["swish", "tanh", "relu"])
    aggregation = sample(["SegmentedMean", "SegmentedMax", "SegmentedMeanMax"])
    quant = sample([0.2, 0.5, 0.75, 0.9])
    loss = sample(["arcface_loss", "arcface_triplet_loss"])
    k = sample(2:10)
    return (hdim=hdim, odim=odim, activation=activation, aggregation=aggregation, quant=quant, loss=loss, k=k)
end

function experiment(dataset::String, parameters, seed, ratios, max_train_time)
    @info "Starting loop for seed no. $seed."

    # load data
    data, labels = load_multiclass(dataset)
    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data, labels; ratios=ratios, seed=seed)
    classes = sort(unique(yk))
    n = c = length(classes)
    Xval, yval, _Xu, _yu = validation_data(yk, Xu, yu, seed, classes)
    Xu, yu = _Xu, _yu
    _Xk, _yk = Xk, yk

    if length(yk)/c < parameters.k
        parameters = merge(parameters, (k=round(Int, length(yk)/c),))
    end

    @info "Data loaded, split and prepared."

    best_models = []
    best_val_losses = []

    for i in 1:5
        @info "Training iteration $i with $(length(yk)) training samples"
        Xk, yk, Xu, yu, model, ls = train_knn(Xk, yk, Xval, yval, Xu, yu, parameters; max_train_time = max_train_time/5)
        push!(best_models, model)
        push!(best_val_losses, ls)
    end

    # find the best model
    ix = findmin(best_val_losses)[2]
    model = best_models[ix][1]
    function predict_labels(model, k, Xk, yk, X, y)
        enc, enc_test = model(Xk), model(X)
        DM = pairwise(Euclidean(), enc_test, enc)
        ypred, acc = dist_knn(k, DM, yk, y)
    end
    accuracy(X, y) = predict_labels(model, parameters.k, _Xk, _yk, X, y)[2]

    ####################
    ### Save results ###
    ####################

    # accuracy results
    train_acc = accuracy(_Xk, _yk)      # known labels
    val_acc = accuracy(Xval, yval)    # validation - used for hyperparameter choice
    test_acc = accuracy(Xt, yt)       # test data - this is the reference accuracy of model quality

    # confusion matrix on test data
    cm, df = confusion_matrix(classes, yt, predict_labels(model, parameters.k, _Xk, _yk, Xt, yt)[1])

    results = Dict(
        :modelname => "self_arcface",
        :parameters => parameters,
        :train_acc => train_acc,
        :val_acc => val_acc,
        :test_acc => test_acc,
        :model => model,
        :CM => (cm, df),
        :seed => seed, 
        :r => r,
    )
    @info "Results calculated, saving..."

    nm = savename(savename(parameters), results, "bson")
    safesave(datadir("experiments", "MIProblems", dataset, "self_arcface", "seed=$seed", nm), results)
end

max_train_time = 60*30

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