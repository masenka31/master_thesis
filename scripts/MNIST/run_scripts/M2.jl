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
function check_params(savepath, parameters)
    s = readdir(savepath)
    cap = map(si -> match(r"known_acc=[0-9\.]*_seed=[0-9]_(.*)\.bson", si).captures[1], s)
    n = savename(parameters)
    any(cap .== n) ? (return false) : (return true)
end


# sample model parameters
function sample_params()
    hdim = sample([16,32,64])           # hidden dimension
    zdim = sample([2,4,8,16])           # latent dimension
    bdim = sample([2,4,8,16])           # the dimension ob output of the HMill model
    batchsize = sample([64, 128, 256])
    agg = sample(["SegmentedMean", "SegmentedMax", "SegmentedMeanMax"])   # HMill aggregation function
    activation = sample(["swish", "relu", "tanh"])                  # activation function
    type = sample([:vanilla, :dense, :simple])
    # α = sample([0.1f0, 0.05f0, 0.01f0])
    α = sample([1f0, 10f0, 100f0])
    return parameters = (hdim = hdim, zdim = zdim, bdim = bdim, batchsize = batchsize, aggregation = agg, activation = activation, type = type, α = α)
end

# function to get validation data and return new unknown data
function validation_data(yk, Xu, yu, seed, classes)
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    c = length(classes)
    n = round(Int, length(yk) / c)
    N = length(yu)

    ik = []
    for i in 1:c
        avail_ix = (1:N)[yu .== classes[i]]
        ix = sample(avail_ix, n)
        push!(ik, ix)
    end
    ik = shuffle(vcat(ik...))
    ileft = setdiff(1:N, ik)

    x, y = reindex(Xu, ik), yu[ik]
    new_xu, new_yu = reindex(Xu, ileft), yu[ileft]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return x, y, new_xu, new_yu
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

    lknown(xk, y) = master_thesis.Models.loss_known_bag_Chamfer(model, xk, y, c)
    lunknown(xu) = master_thesis.Models.loss_unknown_Chamfer(model, xu, c)

    # reconstruction loss - known + unknown
    function loss_rec(Xk, yk, Xu)
        l_known = mean(lknown.(Xk, yk))
        l_unknown = mean(lunknown.(Xu))
        return l_known + l_unknown
    end

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

    # optimizer and training parameters
    opt = ADAM()
    ps = Flux.params(model)
    global max_accuracy = 0
    global best_model = deepcopy(model)
    @info "Best model = $best_model."

    @info "Starting training with parameters $(parameters)..."
    start_time = time()

    while time() - start_time < max_train_time

        b = map(i -> minibatch(), 1:5)
        Flux.train!(lossf, ps, b, opt)

        if isnan(lossf(minibatch()...))
            @info "Loss in NaN, stopped training, moving on..."
            break
        end

        # @show accuracy(Xt, yt)
        @show accuracy(Xk, yk)
        @show a = accuracy(Xval, yval)
        if a >= max_accuracy
            global max_accuracy = a
            global best_model = deepcopy(model)
        end
    end

    @info "Max accuracy = $max_accuracy."
    @info "Best model = $best_model."

    # predict_label(X) = Flux.onecold(model.bagmodel(X), classes)
    predict_label(X) = Flux.onecold(best_model.bagmodel(X), classes)

    # cm, df = confusion_matrix(classes, Xk, yk, predict_label)
    cm, df = confusion_matrix(classes, Xt, yt, predict_label)

    # accuracy of the best model
    best_accuracy(X, y) = accuracy(best_model, X, y, classes)
    ak = best_accuracy(Xk, yk)
    au = best_accuracy(Xu, yu)
    at = best_accuracy(Xt, yt)
    aval = best_accuracy(Xval, yval)

    # chamfer score
    chamfer_label(X) = chamfer_score(best_model, X, classes)
    chamfer_label2(X) = map(i -> chamfer_label(X[i]), 1:nobs(X))
    chamfer_accuracy(X, y) = mean(chamfer_label2(X) .== y)
    chk = mean(i -> chamfer_accuracy(Xk, yk), 1:10)
    ch_val = mean(i -> chamfer_accuracy(Xval, yval), 1:10)
    ch_test = mean(i -> chamfer_accuracy(reindex(Xt, 1:1000), yt[1:1000]), 1:10)

    @info "Results calculated."

    n = savename("known_acc=$(round(max_accuracy, digits=3))_seed=$seed", parameters, "bson")
    results = Dict(
        :parameters => parameters,
        :seed => seed,
        :full => full,
        :r => r,
        :known_acc => ak,
        :unknown_acc => au,
        :val_acc => aval,
        :test_acc => at,
        :chamfer_known => chk,
        :chamfer_val => ch_val,
        :chamfer_test => ch_test,
        :CM => (cm, df),
        :modelname => "M2",
        :model => best_model,
    )

    safesave(datadir("experiments", "MNIST", "M2", n), results)
    @info "Results for seed no. $seed saved."
end

savepath = datadir("experiments", "MNIST", "M2")
ispath(savepath) ? nothing : mkdir(savepath)

# for k in 1:500
    parameters = sample_params()
    # if check_params(savepath, parameters)
        for seed in 1:5
            train_and_save(data, parameters, seed, ratios, full, max_train_time)
        end
    # end
# end