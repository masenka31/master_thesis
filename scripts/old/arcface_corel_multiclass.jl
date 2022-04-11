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

dataset = "animals"
# dataset = "corel"
data, labels = load_multiclass(dataset)

# IF COREL: possible to filter data to have only a few classes
classes = ["Beach", "Sunset", "Buses","Cars","Skiing","Mountains"]
b = map(x -> any(x .== classes), labels)

# prepare data in semisupervised setting
r = 0.1
ratios = (r, 0.5-r, 0.5)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data, labels; ratios=ratios, seed=1)

classes = sort(unique(yk))
n = c = length(classes)

# and encode labels to onehot
Xtrain = Xk
ytrain = yk
yoh_train = Flux.onehotbatch(ytrain, classes)

using Flux: kaiming_uniform
normalizeW(W::AbstractMatrix) = W ./ sqrt.(sum(W .^2, dims=2))
normalizex(x::AbstractVector) = x ./ sqrt(sum(abs2,x))
normalizex(x::AbstractMatrix) = x ./ sqrt.(sum(x .^ 2, dims=1))
normalizes(x::AbstractVector, s::Real) = x ./ sqrt(sum(abs2,x)) * s
normalizes(x::AbstractMatrix, s::Real) = x ./ sqrt.(sum(x .^ 2, dims=1)) .* s

# function arcface_loss(x, y)
#     x_hat = feature_model(x)
#     logit = hardtanh.(normalizeW(W) * normalizes(x_hat, s))
#     θ = acos.(logit)
#     addmargin = cos.(θ .+ m*y)
#     scaled = s .* addmargin
#     Flux.logitcrossentropy(scaled, y)
# end

# emb_model = Chain(mill_model, Mill.data)
function arcface_loss(x, y)
    x_hat = feature_model(x)
    logit = hardtanh.(normalizeW(W) * normalizex(x_hat))   # this is cos θⱼ
    θ = acos.(logit)                                       # this is θⱼ
    addmargin = cos.(θ .+ m*y)                             # there we get cos θⱼ with added margin to θ(yᵢ)
    scaled = s .* addmargin
    Flux.logitcrossentropy(scaled, y)
end

# function arcface_triplet_loss(x, y, labels)
#     x_hat = feature_model(x)
#     logit = hardtanh.(normalizeW(W) * normalizes(x_hat, s))
#     θ = acos.(logit)
#     addmargin = cos.(θ .+ m*y)
#     scaled = s .* addmargin

#     trl = ClusterLosses.loss(Triplet(), SqEuclidean(), x_hat, labels)

#     Flux.logitcrossentropy(scaled, y) + trl
# end
function arcface_triplet_loss(x, y, labels)
    x_hat = feature_model(x)
    logit = hardtanh.(normalizeW(W) * normalizex(x_hat))
    θ = acos.(logit)
    addmargin = cos.(θ .+ m*y)
    scaled = s .* addmargin

    trl = ClusterLosses.loss(Triplet(), SqEuclidean(), x_hat, labels)

    Flux.logitcrossentropy(scaled, y) + trl
end

function arcface_embedding(x)
    x_hat = feature_model(x)
    normalizeW(W) * normalizes(x_hat, s)
end

s = 3f0
m = Float32(π/2/c)

hdim = 32
odim = 2
activation = swish
aggregation = BagCount ∘ SegmentedMeanMax
mdim = 32

mill_model = reflectinmodel(
        Xtrain[1],
        k -> Dense(k, mdim, activation),
        d -> aggregation(d)
    )
feature_model = Chain(mill_model, Mill.data, Dense(mdim, hdim, activation), Dense(mdim, hdim, activation), Dense(hdim, odim))
W = kaiming_uniform(c,odim)
Winit = deepcopy(W)

ps = Flux.params(W, feature_model)
opt = ADAM(0.005)

best_model = deepcopy([feature_model, W])
best_loss = Inf

using Base.Iterators: repeated
for epoch in 1:100
    @info epoch
    # Flux.train!(arcface_loss, ps, repeated((Xtrain, yoh_train), 5), opt)
    # l = arcface_loss(Xtrain, yoh_train)
    Flux.train!(arcface_triplet_loss, ps, repeated((Xtrain, yoh_train, ytrain), 1), opt)
    l = arcface_triplet_loss(Xtrain, yoh_train, ytrain)
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
scatter2(normalizes(enc, s), zcolor=encode(ytrain, classes), aspect_ratio=:equal)
scatter3(normalizes(enc, s), zcolor=encode(ytrain, classes), camera=(20,30))

enc_test = feature_model(Xt)
scatter2(enc_test, zcolor=encode(yt, classes))
scatter2(normalizes(enc_test, s), zcolor=encode(yt, classes), aspect_ratio=:equal)
scatter3(normalizes(enc_test, s), zcolor=encode(yt, classes), camera=(30,30))

# look at the encoding train and test
scatter2(enc, zcolor=encode(ytrain, classes),marker=:circle, label="train",ms=7);
scatter2!(enc_test, zcolor=encode(yt, classes),marker=:square, label="test",ms=3)

scatter3(normalizes(enc, s), zcolor=encode(ytrain, classes), camera=(30,30), label="train",ms=7)
scatter3!(normalizes(enc_test, s), zcolor=encode(yt, classes), camera=(30,30), marker=:square, label="test")

# umap embeddings
model = UMAP_(enc, 2, n_neighbors=5)
emb = transform(model, enc)
scatter2(emb, zcolor=encode(ytrain, classes))

emb_test = transform(model, enc_test)
scatter2(emb_test, zcolor=encode(yt, classes))

emb_test2 = umap(enc_test, 2)
scatter2(emb_test2, zcolor=encode(yt, classes))

# knn
using master_thesis: dist_knn
dm = pairwise(Euclidean(), enc_test, enc)
foreach(k -> dist_knn(k, dm, ytrain, yt), 1:10)

dm_emb = pairwise(Euclidean(), emb_test, emb)
foreach(k -> dist_knn(k, dm_emb, ytrain, yt), 1:10)