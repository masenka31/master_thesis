using DrWatson
@quickactivate

using gvma
using Mill, Flux, StatsBase

include(srcdir("init_strain.jl"))
include(scriptsdir("semisup.jl"))

isempty(ARGS) ? (r = 0.01) : (r = parse(Float64, ARGS[1]))
ratios = (r, 0.5-r, 0.5)
checkpath = datadir("experiments", "gvma", "self_classifier", "seed=1")
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
    hdim = sample([16,32,64,128])         # hidden dimension
    batchsize = sample([64, 128, 256])
    agg = sample(["SegmentedMean", "SegmentedMax", "SegmentedMeanMax"])   # HMill aggregation function
    activation = sample(["relu", "swish", "tanh"])
    threshold = sample([0.9, 0.95, 0.99, 0.999])
    return (hdim=hdim, batchsize=batchsize, agg=agg, activation=activation, threshold=threshold)
end

function experiment(X, y, p, seed, ratios, r, max_train_time)
    @info "Starting loop for seed no. $seed."

    # split data
    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(X, y; ratios=ratios, seed=seed)
    countmap(yk)
    classes = sort(unique(yk))
    n = c = length(classes)
    Xval, yval, _Xu, _yu = validation_data(yk, Xu, yu, seed, classes)
    Xu, yu = _Xu, _yu
    _Xk, _yk = Xk, yk

    # and encode labels to onehot
    hdim, batchsize, threshold = p.hdim, p.batchsize, p.threshold
    activation = eval(Symbol(p.activation))
    aggregation = eval(Symbol(p.agg))
    @info "Data loaded, split and prepared."

    best_models = []
    best_val_accs = []
    ks = [7, 14, 21, 28, 35]

    for i in 1:5
        @info "Training iteration $i with $(length(yk)) training samples"
        Xk, yk, Xu, yu, model, acc = train_classifier(hdim, batchsize, activation, aggregation, threshold, Xk, yk, Xval, yval, Xu, yu; k=ks[i], max_train_time = max_train_time/5)
        push!(best_models, model)
        push!(best_val_accs, acc)
    end

    # find the best model
    ix = findmax(best_val_accs)[2]
    model = best_models[ix];
    accuracy(x, y) = mean(Flux.onecold(model(x), classes) .== y)
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
        :modelname => "self_classifier",
        :parameters => p,
        :train_acc => train_acc,
        :val_acc => val_acc,
        :test_acc => test_acc,
        :model => model,
        :CM => (cm, df),
        :seed => seed, 
        :r => r,
    )
    @info "Results calculated, saving..."

    nm = savename(savename(p), results, "bson")
    safesave(datadir("experiments", "gvma", "self_classifier", "seed=$seed", nm), results)
end

max_train_time = 60*60

for k in 1:500
    parameters = sample_params()
    if check_params(checkpath, parameters, r)
        Threads.@threads for seed in 1:15
            experiment(X, y, parameters, seed, ratios, r, max_train_time)
        end
        break
    else
        @info "Parameters already present, trying new ones."
    end
end
