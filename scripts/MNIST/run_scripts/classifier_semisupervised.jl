using DrWatson
@quickactivate

using master_thesis
using master_thesis: reindex, seqids2bags
using master_thesis: encode

using Flux, Mill
using Random, StatsBase
using DataFrames

# prerequisities
include(srcdir("point_cloud.jl"))
include(scriptsdir("MNIST", "semisup.jl"))

# args
r = parse(Float64, ARGS[1])     # controls ratio of known labels
ratios = (r, 0.5-r, 0.5)        # get the ratios
full = parse(Bool, ARGS[2])

# sample model parameters
function sample_params()
    hdim = sample([16,32,64,128])         # hidden dimension
    batchsize = sample([64, 128, 256])
    agg = sample(["SegmentedMean", "SegmentedMax", "SegmentedMeanMax"])   # HMill aggregation function
    activation = sample(["relu", "swish", "tanh"])
    return hdim, batchsize, agg, activation
end

# load dataset
data = load_mnist_point_cloud()

# define training time -- less train time for smaller number of known data
if r == 0.002
    max_train_time = 60*45
elseif r == 0.01
    max_train_time = 60*90
elseif r == 0.05
    max_train_time = 60*180
end

# model parameters
parameters = sample_params()

function train_and_save(data, parameters, seed, ratios, full, max_train_time)
    @info "Starting loop for seed no. $seed."

    # get parameters
    hdim, batchsize, agg_string, activation_string = parameters
    activation = eval(Symbol(activation_string))
    aggregation = eval(Symbol(agg_string))
    parameters = (hdim = hdim, batchsize = batchsize, agg = agg_string, activation = activation_string)

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

    # prepare data
    Xtrain = Xk
    ytrain = yk
    yoh_train = Flux.onehotbatch(ytrain, classes)

    @info "Data loaded, split and prepared."

    best_models = []
    best_val_accs = []
    ks = [7, 14, 21, 28, 35]

    for i in 1:1
        @show length(yk)
        Xk, yk, Xu, yu, model, acc = train_classifier(hdim, batchsize, activation, aggregation, Xk, yk, Xval, yval, Xu, yu; k=ks[i], max_train_time = max_train_time/5)
        Xtrain = Xk
        ytrain = yk
        yoh_train = Flux.onehotbatch(ytrain, classes)
        push!(best_models, model)
        push!(best_val_accs, acc)
    end

    # find the best model
    ix = findmax(best_val_accs)[2]
    model = best_models[ix]
    accuracy(x::BagNode, y) = round(mean(classes[Flux.onecold(model(x))] .== y), digits=4)
    predict_label(X) = Flux.onecold(model(X), classes)

    ####################
    ### Save results ###
    ####################

    # accuracy results
    train_acc = accuracy(_Xk, _yk)      # known labels
    val_acc = accuracy(Xval, yval)    # validation - used for hyperparameter choice
    test_acc = accuracy(Xt, yt)       # test data - this is the reference accuracy of model quality

    # confusion matrix on test data
    cm, df = confusion_matrix(classes, Xt, yt, predict_label)

    results = Dict(
        :modelname => "kNN_semisupervised_classifier",
        :parameters => parameters,
        :train_acc => train_acc,
        :val_acc => val_acc,
        :test_acc => test_acc,
        :model => model,
        :CM => (cm, df),
        :seed => seed, 
        :r => r,
        :full => full
    )
    @info "Results calculated, saving..."

    nm = savename(savename(parameters), results, "bson")
    safesave(datadir("experiments", "MNIST", "kNN_semisupervised_classifier", nm), results)
end

for seed in 1:5
    train_and_save(data, parameters, seed, ratios, full, max_train_time)
end