using DrWatson
@quickactivate

using master_thesis
using master_thesis.Models
using master_thesis: reindex, seqids2bags, encode

using Flux, Mill
using Random, StatsBase

# prerequisities
include(srcdir("point_cloud.jl"))

# args
r = parse(Float64, ARGS[1])     # controls ratio of known labels
ratios = (r, 0.5-r, 0.5)        # get the ratios
full = parse(Bool, ARGS[2])

# sample model parameters
function sample_params()
    hdim = sample([16,32,64,128])         # hidden dimension
    bdim = sample([2,4,8,16,32])
    zdim = sample([2,4,8,16,32])
    batchsize = sample([64, 128, 256])
    agg = sample(["SegmentedMean", "SegmentedMax", "SegmentedMeanMax"])   # HMill aggregation function
    activation = sample(["relu", "swish", "tanh"])
    α = sample([0.01f0, 0.1f0, 1f0, 10f0])
    return (hdim=hdim, bdim=bdim, zdim=zdim, batchsize=batchsize, agg=agg, activation=activation, α=α)
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
    max_train_time = 60*60*6
else
    max_train_time = 60*60*4
end

function train_and_save(data, parameters, seed, ratios, full, max_train_time)
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

    # get validation data
    Xval, yval, Xu, yu = validation_data(yk, Xu, yu, seed, classes)

    @info "Data loaded, split and prepared."

    # construct model
    model = statistician_constructor(Xk, c; parameters...)
    activation = eval(Symbol(parameters.activation))
    hdim, bdim = parameters.hdim, parameters.bdim
    classifier = Chain(model.instance_encoder, Dense(bdim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, c))

    accuracy(X, y) = mean(Flux.onecold(classifier(X), classes) .== y)
    @show accuracy(Xk, yk)

    # encode labels to 1:c
    ye = encode(yk, classes)

    batchsize = parameters.batchsize
    function minibatch()
        kix = sample(1:nobs(Xk), batchsize)
        uix = sample(1:nobs(Xu), batchsize)
    
        xk, y = reindex(Xk, kix), ye[kix]
        xu = reindex(Xu, uix)
        return xk, y, xu
    end

    lknown(Xb, y) = chamfer_loss(model, Xb, y, c)

    function loss_unknown(statistician, classifier, Xb, c)
        l = map(y -> chamfer_loss(statistician, Xb, y, c), 1:c)
        prob = softmax(classifier(Xb))

        ls = sum(prob .* l)
        e =-  mean(entropy(prob))
        return ls + e
    end
    lunknown(Xb) = loss_unknown(model, classifier, Xb, c)

    N = size(Xk.data.data, 2) # nobs(Xk)
    lclass(Xb, y) = parameters.α * N * Flux.logitcrossentropy(classifier(Xb), Flux.onehotbatch(y, 1:c))

    function lossf(Xk, yk, Xu)
        nk = nobs(Xk)
        bk = Flux.Zygote.@ignore [Xk[i] for i in 1:nk]

        nu = nobs(Xu)
        bu = Flux.Zygote.@ignore [Xu[i] for i in 1:nu]

        lk = mean(lknown.(bk, yk))
        lu = mean(lunknown.(bu))
        lc = lclass(Xk, yk)

        return lk + lu + lc
    end

    # optimizer and training parameters
    opt = ADAM()
    ps = Flux.params(model, classifier)
    best_acc_train = 0
    best_acc_val = 0
    best_model = (model = deepcopy(model), classifier = deepcopy(classifier))
    patience = 0
    max_patience = 200

    @info "Starting training with parameters $(parameters)..."
    
    start_time = time()
    while time() - start_time < max_train_time

        b = map(i -> minibatch(), 1:2)
        Flux.train!(lossf, ps, b, opt)

        if isnan(lossf(minibatch()...))
            @info "Loss in NaN, stopped training, moving on..."
            break
        end

        @show at = accuracy(Xk, yk)
        @show av = accuracy(Xval, yval)
        if (av >= best_acc_val) && (at >= best_acc_train)
            best_acc_train = at
            best_acc_val = av
            best_model = (model = deepcopy(model), classifier = deepcopy(classifier))
            patience = 0
        else
            patience += 1
            if patience > max_patience
                @info "Patience exceeded, stopped trainig."
                break
            end
        end
    end

    @info "Training finished."

    predict_label(X) = Flux.onecold(best_model.classifier(X), classes)
    cm, df = confusion_matrix(classes, Xt, yt, predict_label)

    # accuracy of the best model
    best_accuracy(X, y) = mean(Flux.onecold(best_model.classifier(X), classes) .== y)
    ak = best_accuracy(Xk, yk)
    au = best_accuracy(Xu, yu)
    at = best_accuracy(Xt, yt)
    aval = best_accuracy(Xval, yval)

    @info "Results calculated."
    
    results = Dict(
        :parameters => parameters,
        :seed => seed,
        :full => full,
        :r => r,
        :train_acc => ak,
        :unknown_acc => au,
        :val_acc => aval,
        :test_acc => at,
        :CM => (cm, df),
        :modelname => "statistician",
        :model => best_model,
    )

    n = savename(savename(parameters), results, "bson")
    safesave(datadir("experiments", "MNIST", "statistician", "seed=$seed", n), results)
    @info "Results for seed no. $seed saved."
end

checkpath = datadir("experiments", "MNIST", "statistician", "seed=1")
mkpath(checkpath)

for k in 1:100
    parameters = sample_params()
    if check_params(checkpath, parameters, r, full)
        @info "Parameters checked."
        Threads.@threads for seed in 1:5
            train_and_save(data, parameters, seed, ratios, full, max_train_time)
        end
        break
    else
        @info "Parameters already used, trying new ones..."
    end
end