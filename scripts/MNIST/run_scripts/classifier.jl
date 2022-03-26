using DrWatson
@quickactivate

using master_thesis
using master_thesis: reindex, seqids2bags
using master_thesis: encode

using Flux, Mill
using Random, StatsBase

# prerequisities
include(srcdir("point_cloud.jl"))
project_data(X::AbstractBagNode) = Mill.data(Mill.data(X))

# args
seed = parse(Int, ARGS[1])      # controls data split
r = parse(Float64, ARGS[2])     # controls ratio of known labels
ratios = (r, 0.5-r, 0.5)        # get the ratios
activation_string = ARGS[3]
activation = eval(Symbol(activation_string))
full = parse(Bool, ARGS[4])

# sample model parameters
function sample_params()
    hdim = sample([8,16,32,64])         # hidden dimension
    ldim = sample([2,4,8,16])           # latent dimension (last layer before softmax layer)
    batchsize = sample([64, 128, 256])
    agg = sample([SegmentedMean, SegmentedMax, SegmentedMeanMax])   # HMill aggregation function
    return hdim, ldim, batchsize, agg
end

# load MNIST data
data = load_mnist_point_cloud()

if full
    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.bag_labels; ratios=ratios)
else
    # hardcode to only get 4 predefined numbers
    b = map(x -> any(x .== [0,1,3,4]), data.bag_labels)
    filt_data, filt_labels = reindex(data.data, b), data.bag_labels[b]
    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(filt_data, filt_labels; ratios=ratios)
end

# global parameters
classes = sort(unique(yk))
n = c = length(classes)

# model parameters
hdim, ldim, batchsize, agg = sample_params()

# prepare data
Xtrain = Xk
ytrain = yk
yoh_train = Flux.onehotbatch(ytrain, classes)

function validation_data(yk, Xu, yu, seed)
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

    x, y = reindex(Xu, ik), yu[ik]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return x, y
end

Xval, yval = validation_data(yk, Xu, yu, seed)

@info "Data loaded, split and prepared."

# create the model
mill_model = reflectinmodel(
    Xtrain,
    d -> Dense(d, hdim, activation),
    SegmentedMeanMax
)
model = Chain(
        mill_model, Mill.data,
        Dense(hdim, hdim, activation), Dense(hdim, hdim, activation),
        Dense(hdim, ldim), Dense(ldim, n)
)

# create loss and accuracy functions
loss(x, y) = Flux.logitcrossentropy(model(x), y)
accuracy(x, y) = round(mean(classes[Flux.onecold(model(x))] .== y), digits=4)
opt = ADAM()

# minibatching
function minibatch()
    ix = sample(1:nobs(Xk), batchsize)
    xb = reindex(Xk, ix)
    yb = yoh_train[:, ix]
    xb, yb
end

# define training time -- less train time for smaller number of known data
if r == 0.002
    max_train_time = 60*30
elseif r == 0.01
    max_train_time = 60*60
elseif r == 0.05
    max_train_time = 60*120
end

@info "Starting training with parameters $(parameters)..."
start_time = time()

while time() - start_time < max_train_time
    batches = map(_ -> minibatch(), 1:10)
    Flux.train!(loss, Flux.params(model), batches, opt)
    @info "Batch loss = $(mean(map(x -> loss(x...), batches)))"
    @show accuracy(Xtrain, ytrain)
    @show accuracy(Xval, yval)
end

####################
### Save results ###
####################

parameters = (hdim = hdim, ldim = ldim, batchsize = batchsize, agg = agg)

# accuracy results
train_acc = accuracy(Xk, yk)      # known labels
val_acc = accuracy(Xval, yval)    # validation - used for hyperparameter choice
test_acc = accuracy(Xt, yt)       # test data - this is the reference accuracy of model quality

# confusion matrix on test data
predict_label(X) = Flux.onecold(model(X), classes)
cm, df = confusion_matrix(classes, Xt, yt, predict_label)

results = Dict(
    :modelname => "classifier",
    :parameters => parameters,
    :train_acc => train_acc,
    :val_acc => val_acc,
    :test_acc => test_acc,
    :model => model,
    :CM => (cm, df),
    :seed => seed, 
    :r => r,
    :activation => ARGS[3],
    :full => full
)

nm = savename(savename(parameters), results, "bson")
safesave(datadir("experiments", "MNIST", "classifier", nm), results)