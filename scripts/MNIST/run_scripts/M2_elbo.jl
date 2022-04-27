using DrWatson
@quickactivate
using master_thesis
using master_thesis.Models
using Distributions, DistributionsAD
using ConditionalDists
using Flux

using StatsBase, Random
using LinearAlgebra
using Distances
using Base.Iterators: repeated

using Mill

using master_thesis: reindex, seqids2bags, encode
using master_thesis.Models: chamfer_score

include(srcdir("point_cloud.jl"))

# args
r = parse(Float64, ARGS[1])     # controls ratio of known labels
ratios = (r, 0.5-r, 0.5)        # get the ratios
full = parse(Bool, ARGS[2])
# for now, just testing
# seed = 1

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

# sample model parameters
function sample_params()
    hdim = sample([16,32,64])           # hidden dimension
    zdim = sample([4,8,16])           # latent dimension
    bdim = sample([4,8,16])           # the dimension ob output of the HMill model
    batchsize = sample([64, 128, 256])
    agg = sample([SegmentedMean, SegmentedMax, SegmentedMeanMax])   # HMill aggregation function
    activation = sample(["swish", "relu", "tanh"])                  # activation function
    type = sample([:vanilla, :dense])
    α = sample([1f0, 0.1f0, 10.f0, 0.01f0])
    return parameters = (hdim = hdim, zdim = zdim, bdim = bdim, batchsize = batchsize, aggregation = agg, activation = activation, type = type, α = α)
end

# load MNIST data
data = load_mnist_point_cloud()

# training time
if full
    max_train_time = 60*60*5
else
    max_train_time = 60*60*4
end

function train_and_save(data, parameters, seed, ratios, full, max_train_time)
    @info "Starting loop for seed no. $seed."

    # model parameters
    # hdim, zdim, bdim, batchsize, agg, activation, type, α = pvec
    # parameters = (hdim = hdim, zdim = zdim, bdim = bdim, batchsize = batchsize, aggregation = agg, activation = activation, type = type, α = α)

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
    model = M2_bag_constructor(Xk, c; parameters...)

    function accuracy(model::M2BagModel, X, y, classes)
        ynew = Flux.onecold(model.bagmodel(X), classes)
        mean(ynew .== y)
    end
    accuracy(X, y) = accuracy(model, X, y, classes)
    @info  "Parameters $(parameters)"
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

    lknown(xk, y) = master_thesis.Models.loss_known_bag(model, xk, y, c)
    lunknown(xu) = master_thesis.Models.loss_unknown(model, xu, c)

    # reconstruction loss - known + unknown
    function loss_rec(Xk, yk, Xu)
        l_known = mean(lknown.(Xk, yk))
        l_unknown = mean(lunknown.(Xu))
        return l_known + l_unknown
    end
    loss_rec_known(Xk, yk) = mean(lknown.(Xk, yk))

    N = size(project_data(Xk), 2)
    lclass(x, y) = loss_classification_crossentropy(model, x, y, c) * parameters.α * N

    # now we should be able to dispatch over bags and labels
    function lossf(Xk, yk, Xu)
        nk = nobs(Xk)
        bk = Flux.Zygote.@ignore [Xk[i] for i in 1:nk]

        nu = nobs(Xu)
        bu = Flux.Zygote.@ignore [Xu[i] for i in 1:nu]
        
        lr = loss_rec(bk, yk, bu)
        lc = lclass(Xk, yk)
        return lr + lc
    end
    @show lossf(minibatch()...)

    function loss_warmup(Xk, yk, Xu)
        nk = nobs(Xk)
        bk = Flux.Zygote.@ignore [Xk[i] for i in 1:nk]
        lk = loss_rec_known(bk, yk)
        ce = lclass(Xk, yk)
        return lk + ce
    end
    @show loss_warmup(minibatch()...)

    # optimizer and training parameters
    opt = ADAM()
    ps = Flux.params(model)
    best_acc_train = 0
    best_acc_val = 0
    best_model = deepcopy(model)
    patience = 0
    max_patience = 300

    @info "Starting with one-hour warm-up phase"
    start_time = time()
    while time() - start_time < 60*60
        b = map(i -> minibatch(), 1:1)
        Flux.train!(loss_warmup, ps, b, opt)
        @show a = accuracy(Xk, yk)
    end

    @info "Starting training with parameters $(parameters)..."
    
    start_time = time()
    while time() - start_time < max_train_time-60*60

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
            best_model = deepcopy(model)
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

    predict_label(X) = Flux.onecold(condition(best_model.qy_x, best_model.bagmodel(X)).α, classes)

    # cm, df = confusion_matrix(classes, Xk, yk, predict_label)
    cm, df = confusion_matrix(classes, Xt, yt, predict_label)

    # accuracy of the best model
    best_accuracy(X, y) = accuracy(best_model, X, y, classes)
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
        :modelname => "M2_elbo",
        :model => best_model,
    )

    n = savename(savename(parameters), results, "bson")
    safesave(datadir("experiments", "MNIST", "M2_elbo", "seed=$seed", n), results)
    @info "Results for seed no. $seed saved."
end

checkpath = datadir("experiments", "MNIST", "M2_elbo", "seed=1")
mkpath(checkpath)

for k in 1:100
    parameters = sample_params()
    if check_params(checkpath, parameters, r, full)
        for seed in 1:5
            train_and_save(data, parameters, seed, ratios, full, max_train_time)
        end
        break
    else
        @info "Parameters already used, trying new ones..."
    end
end