using DrWatson
@quickactivate

using gvma
using gvma: encode
using Mill, Flux, StatsBase
using ConditionalDists

include(srcdir("init_strain.jl"))

isempty(ARGS) ? (r = 0.01) : (r = parse(Float64, ARGS[1]))
ratios = (r, 0.5-r, 0.5)
checkpath = datadir("experiments", "gvma", "genmodel", "seed=1")
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

# sample parameters function
function sample_params()
    hdim = sample([16,32,64,128])
    cdim = sample([16,32,64])
    zdim = sample([16,32,64])
    bdim = sample([16,32,64])
    while cdim > hdim
        cdim = sample([16,32,64])
    end
    while zdim > hdim
        zdim = sample([16,32,64])
    end
    while bdim > hdim
        bdim = sample([16,32,64])
    end
    activation = sample(["swish", "tanh", "relu"])
    aggregation = sample(["SegmentedMean", "SegmentedMax", "SegmentedMeanMax"])
    return (hdim=hdim, cdim=cdim, zdim=zdim, bdim=bdim, batchsize=64, activation=activation, aggregation=aggregation)
end

function experiment(X, y, parameters, seed, ratios, r, max_train_time)
    @info "Starting loop for seed no. $seed."

    # split data
    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(X, y; ratios=ratios, seed=seed)
    countmap(yk)
    classes = sort(unique(yk))
    n = c = length(classes)
    Xval, yval, Xu, yu = validation_data(yk, Xu, yu, seed, classes)

    xdata = [Xk[:behavior_summary][i] for i in 1:nobs(Xk)]
    xvdata = [Xval[:behavior_summary][i] for i in 1:nobs(Xval)]
    xudata = [Xu[:behavior_summary][i] for i in 1:nobs(Xu)]
    xtdata = [Xt[:behavior_summary][i] for i in 1:nobs(Xt)]

    @info "Data loaded, split and prepared."

    # get keys to create the models
    ks = keys(X[:behavior_summary])
    lens = mapreduce(k -> size(X[:behavior_summary][k].data.data, 2), vcat, ks)
    fks = ks[lens .> 30]

    # construct model
    model = M2constructor(X[:behavior_summary], fks, c; parameters...)

    # minibatch
    batchsize = parameters.batchsize
    ye = encode(yk, classes)
    function minibatch()
        ixk = sample(1:nobs(Xk), batchsize)
        ixu = sample(1:nobs(Xu), batchsize)
        x = xdata[ixk]
        y = ye[ixk]
        xu = xudata[ixu]
        return x, y, xu
    end
    test_batch = minibatch()

    # define loss function
    α = 0.1f0 * nobs(Xu)
    loss(xk, yk, xu) = semisupervised_loss(model, xk, yk, xu, c, α)
    batch_loss(x, y, xu) = mean(loss.(x, y, xu))
    predictions(X) = Flux.onecold(condition(model.qy_x, model.bagmodel(X)).α, classes)
    # test it
    @show batch_loss(test_batch...)

    # preparation for training
    opt = ADAM()
    ps = Flux.params(model);
    best_tr = 0
    best_val = 0
    best_model = deepcopy(model)

    # train loop
    @info "Starting training with parameteres $parameters..."

    start_time = time()
    while time() - start_time < max_train_time
        batch = minibatch()
        Flux.train!(batch_loss, ps, (batch,), opt)
        @show batch_loss(test_batch...)
        @show tr = mean(predictions(Xk[:behavior_summary]) .== yk)
        @show val = mean(predictions(Xval[:behavior_summary]) .== yval)
        if (val >= best_val) && (tr >= best_tr)
            best_model = deepcopy(model)
            best_val = val
            best_tr = tr
            @info "Validation accuracy improved."
        end
    end

    model = deepcopy(best_model)
    predict_labels(X) = Flux.onecold(condition(model.qy_x, model.bagmodel(X)).α, classes)
    accuracy(X, y) = mean(predict_labels(X) .== y)
    
    ####################
    ### Save results ###
    ####################

    # accuracy results
    train_acc = accuracy(Xk[:behavior_summary], yk)    # known labels
    val_acc = accuracy(Xval[:behavior_summary], yval)    # validation - used for hyperparameter choice
    test_acc = accuracy(Xt[:behavior_summary], yt)       # test data - this is the reference accuracy of model quality

    # confusion matrix on test data
    cm, df = confusion_matrix(classes, yt, predict_labels(Xt[:behavior_summary]))

    results = Dict(
        :modelname => "genmodel",
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
    safesave(datadir("experiments", "gvma", "genmodel", "seed=$seed", nm), results)
end

checkpath = datadir("experiments", "gvma", "genmodel", "seed=1")
mkpath(checkpath)

max_train_time = 60*60*7

for k in 1:10
    parameters = sample_params()
    if check_params(checkpath, parameters, r)
        Threads.@threads for seed in 1:6
            experiment(X, y, parameters, seed, ratios, r, max_train_time)
        end
        break
    else
        @info "Parameters already present, trying new ones."
    end
end