using Revise
using DrWatson
@quickactivate

using BSON, Mill
using Flux
using Flux: @epochs
using Base.Iterators: repeated
using ClusterLosses

using Plots
gr(markerstrokewidth=0, color=:jet)
using master_thesis
using master_thesis: encode

using UMAP, Statistics
using Distances, Clustering

corel = BSON.load(datadir("corel_mill.bson"))
@unpack labels, X, bagids = corel

_data = BagNode(ArrayNode(X), bagids)

_y = []
labelnames = unique(labels)

# create bag labels
for i in 1:2000
    bl = labels[bagids .== i]
    # check that the labels are unique
    if length(unique(bl)) > 1
        @warn "Labels are not the same!"
    end
    push!(_y, bl[1])
end

# filter data to have only a few classes
# classes = sample(unique(_y), 10, replace=false)
classes = ["Beach", "Sunset", "Buses","Cars","Skiing","Mountains"]
b = map(x -> any(x .== classes), _y)
data = _data[b]
y = _y[b]


# Testing on train/test splits
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(data, y, ratio=0.5)
labelnames = unique(y)
c = length(labelnames)
yoh = Flux.onehotbatch(ytrain, labelnames)

using Flux: kaiming_uniform
normalizeW(W::AbstractMatrix) = W ./ sqrt.(sum(W .^2, dims=2))
normalizes(x::AbstractVector, s::Real) = x ./ sqrt(sum(abs2,x)) * s
normalizes(x::AbstractMatrix, s::Real) = x ./ sqrt.(sum(x .^ 2, dims=1)) .* s

function arcface_loss(x, y)
    x_hat = feature_model(x)
    logit = tanh.(normalizeW(W) * normalizes(x_hat, s))
    θ = acos.(logit)
    addmargin = cos.(θ .+ m*y)
    scaled = s .* addmargin
    Flux.logitcrossentropy(scaled, y)
end

function arcface_triplet_loss(x, y, labels)
    x_hat = feature_model(x)
    logit = tanh.(normalizeW(W) * normalizes(x_hat, s))
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
m = 0.25f0

hdim = 32
odim = 3
activation = swish
aggregation = meanmax_aggregation
mdim = 32

mill_model = reflectinmodel(
        Xtrain[1],
        k -> Dense(k, mdim, activation),
        d -> aggregation(d)
    )
feature_model = Chain(mill_model, Mill.data, Dense(mdim, hdim, activation), Dense(mdim, hdim, activation), Dense(hdim, odim))
W = kaiming_uniform(c,odim)

ps = Flux.params(W, feature_model)
opt = ADAM(0.001)

for epoch in 1:1000
    @info epoch
    #Flux.train!(arcface_loss, ps, repeated((Xtrain, yoh), 1), opt)
    Flux.train!(arcface_triplet_loss, ps, repeated((Xtrain, yoh, ytrain), 1), opt)
    l = arcface_triplet_loss(Xtrain, yoh, ytrain)
    @show l
end

enc = feature_model(Xtrain)
scatter2(enc, zcolor=encode(ytrain, labelnames), xlims=(-3,3),ylims=(-3,3))
scatter2(normalizes(enc, s), zcolor=encode(ytrain, labelnames), aspect_ratio=:equal)
scatter3(normalizes(enc, s), zcolor=encode(ytrain, labelnames), camera=(20,30))

enc_test = feature_model(Xtest)
scatter2(enc_test, zcolor=encode(ytest, labelnames))
scatter2(normalizes(enc_test, s), zcolor=encode(ytest, labelnames), aspect_ratio=:equal)
scatter3(normalizes(enc_test, s), zcolor=encode(ytest, labelnames), camera=(30,30))

# look at the encoding train and test
scatter2(enc, zcolor=encode(ytrain, labelnames),marker=:circle, label="train");
scatter2!(enc_test, zcolor=encode(ytest, labelnames),marker=:square, label="test")

scatter3(normalizes(enc, s), zcolor=encode(ytrain, labelnames), camera=(30,30), label="train")
scatter3!(normalizes(enc_test, s), zcolor=encode(ytest, labelnames), camera=(30,30), marker=:square, label="test")

# umap embeddings
model = UMAP_(enc, 2, n_neighbors=15)
emb = transform(model, enc)
scatter2(emb, zcolor=encode(ytrain, labelnames))

emb_test = transform(model, enc_test)
scatter2(emb_test, zcolor=encode(ytest, labelnames))

emb_test2 = umap(enc_test, 2)
scatter2(emb_test2, zcolor=encode(ytest, labelnames))

# knn
using master_thesis: dist_knn
dm = pairwise(Euclidean(), enc_test, enc)
foreach(k -> dist_knn(k, dm, ytrain, ytest), 1:50)

dm_emb = pairwise(Euclidean(), emb_test, emb)
foreach(k -> dist_knn(k, dm_emb, ytrain, ytest), 1:50)


# how about new data?
new_classes = sample(setdiff(unique(_y), classes), 1, replace=false)
b = map(x -> any(x .== new_classes), _y)
newdata = _data[b]
newy = _y[b]

fdata = cat(Xtrain, newdata)
fy = vcat(ytrain, newy)
fclasses = vcat(classes, new_classes)

enc = feature_model(fdata)
scatter2(enc, zcolor=encode(fy, fclasses));
scatter2(normalizes(enc, s), zcolor=encode(fy, fclasses), aspect_ratio=:equal, opacity=0.7);
scatter3(normalizes(enc, s), zcolor=encode(fy, fclasses), camera=(70,30),title=new_classes[1])

emb = umap(enc, 2, n_neighbors=15)
scatter2(emb, zcolor=encode(fy, unique(fy)))