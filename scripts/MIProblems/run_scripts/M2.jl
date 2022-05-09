using DrWatson
@quickactivate

using master_thesis
using master_thesis: dist_knn, encode
using master_thesis.Models
using ConditionalDists, Distributions, DistributionsAD

using Flux
using Distances

include(srcdir("point_cloud.jl"))
include(srcdir("mill_data.jl"))

r = parse(Float64, ARGS[1])
ratios = (r, 0.5-r, 0.5)
max_train_time = 60*15
dataset = ARGS[2]
# dataset = "animals"
checkpath = datadir("experiments", "MIProblems", dataset, "M2", "seed=1")
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
    hdim = sample([16,32,64])           # hidden dimension
    zdim = sample([2,4,8,16])           # latent dimension
    bdim = sample([2,4,8,16])           # the dimension ob output of the HMill model
    batchsize = sample([32, 64])
    agg = sample(["SegmentedMean", "SegmentedMax", "SegmentedMeanMax"])   # HMill aggregation function
    activation = sample(["swish", "relu", "tanh"])                  # activation function
    type = sample([:vanilla, :dense, :simple])
    α = sample([0.1f0, 1f0, 10f0])
    return parameters = (hdim = hdim, zdim = zdim, bdim = bdim, batchsize = batchsize, aggregation = agg, activation = activation, type = type, α = α)
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

function experiment(dataset::String, parameters, seed, ratios, max_train_time)
    @info "Starting loop for seed no. $seed."

    data, labels = load_multiclass(dataset)

    Xk, yk, _Xu, _yu, Xt, yt = split_semisupervised_balanced(data, labels; ratios=ratios, seed=seed)
    classes = sort(unique(yk))
    n = c = length(classes)
    Xval, yval, Xu, yu = validation_data(yk, _Xu, _yu, seed, classes)

    @info "Data loaded, split and prepared."

    # construct model
    model = M2_bag_constructor(Xk, c; parameters...)

    # accuracy
    function accuracy(model::M2BagModel, X, y, classes)
        ynew = Flux.onecold(condition(model.qy_x, model.bagmodel(X)).α, classes)
        mean(ynew .== y)
    end
    accuracy(X, y) = accuracy(model, X, y, classes)
    @show accuracy(Xk, yk)

    # encode labels to 1:c
    ye = encode(yk, classes)

    # minibatch
    batchsize = parameters.batchsize
    function minibatch()
        kix = sample(1:nobs(Xk), batchsize)
        uix = sample(1:nobs(Xu), batchsize)

        xk, y = reindex(Xk, kix), ye[kix]
        xu = reindex(Xu, uix)
        return xk, y, xu
    end

    # loss functions
    lknown(xk, y) = master_thesis.Models.loss_known_bag(model, xk, y, c)
    lunknown(xu) = master_thesis.Models.loss_unknown(model, xu, c)

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
    max_accuracy = 0
    best_model = deepcopy(model)

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
        # @show accuracy(Xk, yk)
        a = accuracy(Xk, yk)
        # @show accuracy(Xval, yval)
        if a >= max_accuracy
            max_accuracy = a
            best_model = deepcopy(model)
        end
    end

    @info "Max accuracy = $max_accuracy."

    # predict_label(X) = Flux.onecold(model.bagmodel(X), classes)
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
        :r => r,
        :train_acc => ak,
        :unknown_acc => au,
        :val_acc => aval,
        :test_acc => at,
        :CM => (cm, df),
        :modelname => "M2",
        :model => best_model,
    )

    n = savename(savename(parameters), results, "bson")
    @info "Saving to: $(datadir("experiments", "MIProblems", dataset, "M2", "seed=$seed", n))"
    safesave(datadir("experiments", "MIProblems", dataset, "M2", "seed=$seed", n), results)
    @info "Results for seed no. $seed saved."
end

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