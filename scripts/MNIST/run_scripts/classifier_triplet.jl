using DrWatson
@quickactivate

using master_thesis
using master_thesis: reindex, seqids2bags
using master_thesis: encode

using Flux, Mill
using ClusterLosses
using Random, StatsBase
using Distances

# prerequisities
include(srcdir("point_cloud.jl"))

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
    margin = sample([0.5f0, 1f0])
    α = sample([0.1f0, 1f0])
    return hdim, batchsize, agg, activation, margin, α
end

# check that the combination does not yet exist
function check_params(checkpath, parameters, r, fl)
    files = readdir(checkpath)
    n = savename(parameters)
    b_par = map(f -> occursin(n, f), files)
    b_r = map(f -> occursin("r=$r", f), files)
    b_f = map(f -> occursin("full=$fl", f), files)
    b = b_par .* b_r .* b_f
    # any(b) ? (return false) : (return true)
    any(b) ? false : true
end

# load dataset
data = load_mnist_point_cloud()

# define training time -- less train time for smaller number of known data
if full
    max_train_time = 60*120
else
    max_train_time = 60*60
end

function train_and_save(data, pvec, seed, ratios, full, max_train_time)
    @info "Starting loop for seed no. $seed."

    # get parameters
    hdim, batchsize, agg_string, activation_string, margin, α = pvec
    activation = eval(Symbol(activation_string))
    aggregation = eval(Symbol(agg_string))
    parameters = (hdim = hdim, batchsize = batchsize, agg = agg_string, activation = activation_string, margin=margin, α=α)

    # split dataset
    if full
        Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.bag_labels; ratios=ratios, seed=seed)
    else
        # hardcode to only get 4 predefined numbers
        b = map(x -> any(x .== [0,1,3,4]), data.bag_labels)
        filt_data, filt_labels = reindex(data.data, b), data.bag_labels[b]
        Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(filt_data, filt_labels; ratios=ratios, seed=seed)
    end

    # minibatching
    function minibatch()
        ix = sample(1:nobs(Xk), batchsize)
        xb = reindex(Xk, ix)
        yb = yoh_train[:, ix]
        yl = ytrain[ix]
        xb, yb, yl
    end

    # global parameters
    classes = sort(unique(yk))
    n = c = length(classes)
    Xval, yval, Xu, yu = validation_data(yk, Xu, yu, seed, classes)

    # prepare data
    Xtrain = Xk
    ytrain = yk
    yoh_train = Flux.onehotbatch(ytrain, classes)

    @info "Data loaded, split and prepared."

    # create the model
    mill_model = reflectinmodel(
        Xtrain,
        d -> Dense(d, hdim, activation),
        BagCount ∘ aggregation
    )
    model = Chain(
        mill_model, Mill.data,
        Dense(hdim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, n)
    )

    # create loss and accuracy functions
    function loss(x, yoh, y)
        ce = Flux.logitcrossentropy(model(x), yoh)
        enc = Mill.data(mill_model(x))
        trl = ClusterLosses.loss(Triplet(margin), SqEuclidean(), enc, y)

        return ce + α * trl
    end
    accuracy(x, y) = mean(classes[Flux.onecold(model(x))] .== y)
    best_accuracy(x, y) = mean(Flux.onecold(best_model(x), classes) .== y)
    opt = ADAM()

    best_val_acc = 0
    best_train_acc = 0
    best_model = deepcopy(model)
    patience = 0
    max_patience = 200

    @info "Starting training with parameters $(parameters)..."
    start_time = time()

    while time() - start_time < max_train_time
        batches = map(_ -> minibatch(), 1:10)
        Flux.train!(loss, Flux.params(model), batches, opt)

        a = accuracy(Xval, yval)
        ak = accuracy(Xk, yk)
        if (a > best_val_acc) && (ak >= best_train_acc)
            @show a
            @show accuracy(Xt, yt)
            best_model = deepcopy(model)
            best_train_acc = ak
            best_val_acc = a
            patience = 0
        elseif (a == best_val_acc) && (ak >= best_train_acc)
            best_train_acc = ak
            best_val_acc = a
            best_model = deepcopy(model)
            print(".")
            patience += 1
            if (patience > max_patience) && (best_accuracy(Xk, yk) == 1.0)
                @info "Patience exceeded, training stopped."
                break
            end
        else
            print(".")
            patience += 1
            if (patience > max_patience) && (best_accuracy(Xk, yk) == 1.0)
                @info "Patience exceeded, training stopped."
                break
            end
        end
    end
    @info "Training finished."
    model = deepcopy(best_model)

    ####################
    ### Save results ###
    ####################

    # accuracy results
    train_acc = accuracy(Xk, yk)      # known labels
    val_acc = accuracy(Xval, yval)    # validation - used for hyperparameter choice
    test_acc = accuracy(Xt, yt)       # test data - this is the reference accuracy of model quality

    # confusion matrix on test data
    predict_label(X) = Flux.onecold(model(X), classes)
    cm, df = confusion_matrix(classes, Xt, yt, predict_label)

    results = Dict(
        :modelname => "classifier_triplet",
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
    safesave(datadir("experiments", "MNIST", "classifier_triplet", "seed=$seed", nm), results)
end

checkpath = datadir("experiments", "MNIST", "classifier_triplet", "seed=1")
mkpath(checkpath)

for i in 1:100
    # sample parameters
    pvec = sample_params()
    hdim, batchsize, agg_string, activation_string, margin, α = pvec
    parameters = (hdim = hdim, batchsize = batchsize, agg = agg_string, activation = activation_string, margin=margin, α=α)

    if check_params(checkpath, parameters, r, full)
        @info "Parameters checked."
        for seed in 1:5
            train_and_save(data, pvec, seed, ratios, full, max_train_time)
        end
        break
    else
        @info "Parameters already present, trying new ones..."
    end
end