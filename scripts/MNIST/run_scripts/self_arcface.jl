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
include(scriptsdir("MNIST", "semisup.jl"))

# args
r = parse(Float64, ARGS[1])     # controls ratio of known labels
ratios = (r, 0.5-r, 0.5)        # get the ratios
full = parse(Bool, ARGS[2])

# load dataset
data = load_mnist_point_cloud()

# define training time -- less train time for smaller number of known data
if full
    max_train_time = 60*60*3.5
else
    max_train_time = 60*60*2.5
end

# check that the combination does not yet exist
function check_params(checkpath, parameters, r, full)
    files = readdir(checkpath)
    n = savename(parameters)
    b_par = map(f -> occursin(n, f), files)
    b_r = map(f -> occursin("r=$r", f), files)
    b_f = map(f -> occursin("full=$full", f), files)
    b = b_par .* b_r .* b_f
    # any(b) ? (return false) : (return true)
    any(b) ? false : true
end

function sample_params()
    hdim = sample([16,32,64,128])
    odim = sample([4,8,16,32])
    while odim > hdim
        odim = sample([2,4,8,16])
    end
    activation = sample(["swish", "tanh", "relu"])
    aggregation = sample(["SegmentedMean", "SegmentedMax", "SegmentedMeanMax"])
    quant = sample([0.2, 0.5, 0.75, 0.9])
    loss = sample(["arcface_loss", "arcface_triplet_loss"])
    k = sample(3:3:30)
    return (hdim=hdim, odim=odim, activation=activation, aggregation=aggregation, quant=quant, loss=loss, k=k)
end

function experiment(data, parameters, seed, ratios, full, max_train_time)
    @info "Starting loop for seed no. $seed."

    # split dataset
    if full
        Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.bag_labels; ratios=ratios, seed=seed)
    else
        # hardcode to only get 4 predefined numbers
        b = map(x -> any(x .== [0,1,3,4]), data.bag_labels)
        filt_data, filt_labels = reindex(data.data, b), data.bag_labels[b]
        Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(filt_data, filt_labels; ratios=ratios, seed=seed)
    end

    # global parameters
    classes = sort(unique(yk))
    n = c = length(classes)
    Xval, yval, _Xu, _yu = validation_data(yk, Xu, yu, seed, classes)
    Xu, yu = _Xu, _yu
    _Xk, _yk = Xk, yk

    @info "Data loaded, split and prepared."

    if length(yk)/c < parameters.k
        new_k = round(Int, length(yk)/c)
        parameters = merge(parameters, (k=new_k,))
    end

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
    safesave(datadir("experiments", "MNIST", "self_arcface", "seed=$seed", nm), results)
end

checkpath = datadir("experiments", "MNIST", "self_arcface", "seed=1")
mkpath(checkpath)

for k in 1:500
    parameters = sample_params()
    if check_params(checkpath, parameters, r, full)
        Threads.@threads for seed in 1:5
            experiment(data, parameters, seed, ratios, full, max_train_time)
        end
        break
    else
        @info "Parameters already present, trying new ones."
    end
end