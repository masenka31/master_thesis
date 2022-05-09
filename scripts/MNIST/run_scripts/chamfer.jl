using DrWatson
@quickactivate
using master_thesis
using master_thesis.Models
using master_thesis: reindex, seqids2bags, encode
using Flux, Mill

include(srcdir("point_cloud.jl"))

# args
r = parse(Float64, ARGS[1])     # controls ratio of known labels
ratios = (r, 0.5-r, 0.5)        # get the ratios
full = parse(Bool, ARGS[2])

# sample model parameters
function sample_params()
    hdim = sample([16,32,64,128])          # hidden dimension
    cdim = sample([2,4,8,16,32])           # context dimension
    bdim = sample([2,4,8,16,32])           # bagmodel dimension
    gdim = sample([2,4,8,16,32])           # generator dimension
    batchsize = sample([64, 128, 256])
    agg = sample(["SegmentedMean", "SegmentedMax", "SegmentedMeanMax"])   # HMill aggregation function
    activation = sample(["swish", "relu", "tanh"])                        # activation function
    return parameters = (hdim = hdim, cdim = cdim, bdim = gdim, gdim = gdim, batchsize = batchsize, aggregation = agg, activation = activation)
end

# load MNIST data
data = load_mnist_point_cloud()

# training time
if full
    max_train_time = 60*60*6
else
    max_train_time = 60*60*5
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
    aggregation = eval(Symbol(parameters.aggregation))
    agg = BagCount ∘ aggregation
    m = chamfermodel_constructor(
        Xk, c; cdim=parameters.cdim, hdim=parameters.hdim, bdim=parameters.bdim,
        gdim=parameters.gdim, aggregation=agg, activation=parameters.activation
    )

    # accuracy
    function accuracy(m::ChamferModel, X, y, classes)
        ynew = Flux.onecold(m.classifier(m.bagmodel(X)), classes)
        mean(ynew .== y)
    end
    accuracy(X, y) = accuracy(m, X, y, classes)
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

    # loss functions
    lknown(xk, y) = loss_known(m, xk, y, c)
    lunknown(xu) = loss_unknown(m, xu, c)
    
    function loss_rec(Xk, yk, Xu)
        l_known = mean(lknown.(Xk, yk))
        l_unknown = mean(lunknown.(Xu))
        return l_known + l_unknown
    end
    
    N = nobs(Xk)
    α = 0.1f0
    lclass(x, y) = loss_classification(m, x, y, c) * α * N
    
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
    ps = Flux.params(m)
    max_accuracy = 0
    best_model = deepcopy(m)

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
        a = accuracy(Xval, yval)
        if a >= max_accuracy
            max_accuracy = a
            best_model = deepcopy(m)
        end
    end

    @info "Training finished.\nMax accuracy = $max_accuracy."

    # predict_label(X) = Flux.onecold(model.bagmodel(X), classes)
    predict_label(X) = Flux.onecold(best_model.classifier(best_model.bagmodel(X)), classes)

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
        :known_acc => ak,
        :unknown_acc => au,
        :val_acc => aval,
        :test_acc => at,
        :CM => (cm, df),
        :modelname => "ChamferModel",
        :model => best_model,
    )

    n = savename(parameters)*savename(results)*".bson"
    safesave(datadir("experiments", "MNIST", "ChamferModel", "seed=$seed", n), results)
    @info "Results for seed no. $seed saved."
end

# max_train_time = 60 # for testing
parameters = sample_params()
for seed in 1:5
    train_and_save(data, parameters, seed, ratios, full, max_train_time)
end