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

include(srcdir("point_cloud.jl"))

# args
r = parse(Float64, ARGS[1])     # controls ratio of known labels
ratios = (r, 0.5-r, 0.5)        # get the ratios
full = parse(Bool, ARGS[2])

# sample model parameters
function sample_params()
    hdim = sample([8,16,32,64])         # hidden dimension
    zdim = sample([2,4,8,16])           # latent dimension
    bdim = sample([2,4,8,16])           # the dimension ob output of the HMill model
    batchsize = sample([64, 128, 256])
    agg = sample([SegmentedMean, SegmentedMax, SegmentedMeanMax])   # HMill aggregation function
    activation = sample(["swish", "relu", "tanh"])                  # activation function
    type = sample([:vanilla, :dense, :simple])
    return hdim, zdim, bdim, batchsize, agg, activation, type
end

# load MNIST data
data = load_mnist_point_cloud()

if full
    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.bag_labels; ratios=ratios, seed=seed)
else
    # hardcode to only get 4 predefined numbers
    b = map(x -> any(x .== [0,1,3,4]), data.bag_labels)
    filt_data, filt_labels = reindex(data.data, b), data.bag_labels[b]
    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(filt_data, filt_labels; ratios=ratios, seed=seed)
end

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
    uk = (1:length(yu))

    x, y = reindex(Xu, ik), yu[ik]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return x, y
end

# global parameters
classes = sort(unique(yk))
n = c = length(classes)

# model parameters
hdim, zdim, bdim, batchsize, agg, activation, type = sample_params()
parameters = (hdim = hdim, zdim = zdim, bdim = bdim, batchsize = batchsize, aggregation = agg, activation = activation, type = type)

model = M2_bag_constructor(Xk, c; parameters...)

function accuracy(model::M2BagModel, X, y, classes)
    ynew = Flux.onecold(model.bagmodel(X), classes)
    mean(ynew .== y)
end
accuracy(X, y) = accuracy(model, X, y, classes)

# encode labels to 1:c
ye = encode(yk, classes)
batchsize = 64

function minibatch()
    kix = sample(1:nobs(Xk), batchsize)
    uix = sample(1:nobs(Xu), batchsize)

    xk, y = reindex(Xk, kix), ye[kix]
    xu = reindex(Xu, uix)
    return xk, y, xu
end

N = size(project_data(Xk), 2)
lclass(x, y) = loss_classification_crossentropy(model, x, y, c) * 0.1f0 * N

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

# optimizer and training parameters
opt = ADAM(0.001)
ps = Flux.params(model)
max_accuracy = 0
best_model = deepcopy(model)

# trainign times
max_train_time = 60*10
start_time = time()

while time() - start_time < max_train_time

    b = map(i -> minibatch(), 1:5)
    Flux.train!(lossf, ps, b, opt)

    # @show accuracy(Xt, yt)
    @show a = accuracy(Xk, yk)
    a >= max_accuracy ? (global max_accuracy, best_model = a, deepcopy(model)) : nothing
end