using DrWatson
@quickactivate

using master_thesis
using master_thesis.gvma

include(srcdir("gvma", "init_strain.jl"))

isempty(ARGS) ? (r = 0.01) : (r = parse(Float64, ARGS[1]))
ratios = (r, 0.5-r, 0.5)
checkpath = datadir("experiments", "gvma", "classifier", "seed=1")
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
    return (hdim = hdim, batchsize = batchsize, agg = agg, activation = activation)
end

function experiment(X, y, p, seed, ratios, r, max_train_time)
    @info "Starting loop for seed no. $seed."
    
    # split data
    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(X, y; ratios=ratios, seed=seed)
    countmap(yk)
    classes = sort(unique(yk))
    n = c = length(classes)
    Xval, yval, Xu, yu = validation_data(yk, Xu, yu, seed, classes)

    # and encode labels to onehot
    Xtrain = Xk
    ytrain = yk
    yoh_train = Flux.onehotbatch(ytrain, classes)

    @info "Data prepared and split."

    # parameters
    activation = eval(Symbol(p.activation))
    aggregation = eval(Symbol(p.agg))

    # create a simple classificator model
    mill_model = reflectinmodel(
        # sch, ex,
        Xk[1],
        d -> Dense(d, p.hdim, activation),
        aggregation
    )
    model = Chain(
            mill_model, Dense(p.hdim, p.hdim, activation), Dense(p.hdim, p.hdim, activation), Dense(p.hdim, n)
    );

    loss(x, y) = Flux.logitcrossentropy(model(x), y)
    accuracy(x, y) = mean(Flux.onecold(model(x), classes) .== y)
    best_accuracy(x, y) = mean(Flux.onecold(best_model(x), classes) .== y)

    function minibatch()
        ix = sample(1:nobs(Xk), p.batchsize)
        xb = reindex(Xk, ix)
        yb = yoh_train[:, ix]
        xb, yb
    end

    opt = ADAM()
    best_val_acc = 0
    best_train_acc = 0
    best_model = deepcopy(model);
    patience = 0
    max_patience = 200

    @info "Starting training with parameters $(p)..."

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
    model = deepcopy(best_model);

    ####################
    ### Save results ###
    ####################

    # results only for the best model
    predict_label(X) = Flux.onecold(best_model(X), classes)

    # accuracy results
    train_acc = best_accuracy(Xk, yk)      # known labels
    val_acc = best_accuracy(Xval, yval)    # validation - used for hyperparameter choice
    test_acc = best_accuracy(Xt, yt)       # test data - this is the reference accuracy of model quality

    # confusion matrix on test data
    cm, df = confusion_matrix(classes, Xt, yt, predict_label)

    results = Dict(
        :modelname => "classifier",
        :parameters => p,
        :train_acc => train_acc,
        :val_acc => val_acc,
        :test_acc => test_acc,
        :model => best_model,
        :CM => (cm, df),
        :seed => seed,
        :r => r,
    )
    @info "Results for seed no. $seed calculated, saving..."

    nm = savename(savename(p), results, "bson")
    # @show nm
    safesave(datadir("experiments", "gvma", "classifier", "seed=$seed", nm), results)
    @info "Results saved."
end

max_train_time = 60*20

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