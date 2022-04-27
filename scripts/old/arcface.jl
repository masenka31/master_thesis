using Revise
using DrWatson
@quickactivate

using Flux, Mill
using ClusterLosses

using Plots
ENV["GKSwstype"] = "100"
gr(markerstrokewidth=0, color=:jet, label="")

using master_thesis
using master_thesis: encode

using UMAP, Statistics
using Distances, Clustering

include(srcdir("point_cloud.jl"))
include(srcdir("mill_data.jl"))

dataset = "animals_negative"
data, labels = load_multiclass(dataset)

function sample_params()
    hdim = sample([8,16,32,64])
    odim = sample([2,4,8,16])
    while odim > hdim
        odim = sample([2,4,8,16])
    end
    activation = sample(["swish", "tanh", "relu"])
    aggregation = sample(["SegmentedMean", "SegmentedMax", "SegmentedMeanMax"])
    quant = sample([0.2, 0.5, 0.75, 0.9])
    triplet = sample([true, false])
    k = sample(2:10)
    return (hdim=hdim, odim=odim, activation=activation, aggregation=aggregation, quant=quant, triplet=triplet)
end

# prepare data in semisupervised setting
r = 0.1
ratios = (r, 0.5-r, 0.5)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data, labels; ratios=ratios, seed=1)

classes = sort(unique(yk))
n = c = length(classes)
Xval, yval, Xu, yu = validation_data(yk, Xu, yu, 1, classes)

# and encode labels to onehot
Xtrain = Xk
ytrain = yk
yoh_train = Flux.onehotbatch(ytrain, classes)

function arcface_constructor(Xtrain, c; hdim, odim, activation, aggregation, kwargs...)
    aggregation = eval(Symbol(aggregation))
    activation = eval(Symbol(activation))
    mill_model = reflectinmodel(
            Xtrain[1],
            k -> Dense(k, hdim, activation),
            BagCount ∘ aggregation
    )
    feature_model = Chain(mill_model, Mill.data, Dense(hdim, hdim, activation), Dense(hdim, odim))
    W = Flux.kaiming_uniform(c, odim)
    return feature_model, W
end

# model
feature_model, W = arcface_constructor(Xtrain, c; parameters...)

# parameters
s = 64f0
m = 0.5f0

if parameters.triplet
    @info "Choosing ArcFace with Triplet loss regularization."
    loss(x, y, yk) = arcface_triplet_loss(feature_model, W, m, s, x, y, yk)
else
    @info "Choosing ArcFace."
    loss(x, y, yk) = arcface_loss(feature_model, W, m, s, x, y)
end

loss(Xtrain, yoh_train, yk)

# Flux initialize
ps = Flux.params(W, feature_model)
opt = ADAM(0.005)

best_model = deepcopy([feature_model, W])
best_loss = Inf

using Base.Iterators: repeated
for epoch in 1:100
    @info epoch
    Flux.train!(loss, ps, repeated((Xtrain, yoh_train, ytrain), 5), opt)
    l = loss(Xtrain, yoh_train, ytrain)
    @show l
    if l < best_loss
        best_loss = l
        best_model = deepcopy([feature_model, W])
    end
    if isnan(l)
        break
        @info "Loss is NaN."
    end
end

feature_model, W = best_model

enc = feature_model(Xtrain)
scatter2(enc, zcolor=encode(ytrain, classes))

enc_test = feature_model(Xt)
scatter2(enc_test, zcolor=encode(yt, classes))

# look at the encoding train and test
scatter2(enc, zcolor=encode(ytrain, classes),marker=:circle, label="train",ms=7);
scatter2!(enc_test, zcolor=encode(yt, classes),marker=:square, label="test",ms=3)

# classification
k = 10
classify_unknown_knn(feature_model, Xk, yk, Xu, yu, k, classes; quant=parameters.quant)

using NearestNeighbors
function classify_unknown_knn(model, Xk::T, yk, Xu::T, yu, k::Int, classes; quant=0.2) where T <: AbstractMillNode
    # calculate encodings
    enc = model(Xk)
    enc_unknown = model(Xu)

    # create tree and find k-nearest neighbors for each point in Xu
    tree = BruteTree(enc)
    idxs, dists = knn(tree, enc_unknown, k)

    # calculate quantile mean distance
    md = mean.(dists)
    q = quantile(md, quant)

    # filter only the samples which have k train neighbors from same class
    # and the distance is in the 20% quantile
    bk = map(idx -> length(unique(yk[idx])) == 1, idxs) |> BitVector
    bq = md .< q
    b = bk .* bq
    ixs = (1:nobs(Xu))[b]
    ixs_left = setdiff(1:nobs(Xu), ixs)
    
    Xknew = reindex(Xu, ixs)
    yknew = map(idx -> unique(yk[idx])[1], idxs[ixs])
    
    @info "Accuracy of inferred labels: $(mean(yu[ixs] .== yknew))."

    Xunew = reindex(Xu, ixs_left)
    yunew = yu[ixs_left]

    return Xknew, yknew, Xunew, yunew, ixs
end