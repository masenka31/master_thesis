using DrWatson
@quickactivate

using master_thesis
using Flux
using Distances

include(srcdir("point_cloud.jl"))
include(srcdir("mill_data.jl"))

function load_animals()
    fox = load_mill_data("Fox").normal
    elephant = load_mill_data("Elephant").normal
    tiger = load_mill_data("Tiger").normal

    yf = repeat(["Fox"], nobs(fox))
    ye = repeat(["Elephant"], nobs(elephant))
    yt = repeat(["Tiger"], nobs(tiger))

    return cat(fox, elephant, tiger), vcat(yf, ye, yt)
end

data, labels = load_animals()

r = 0.05
ratios = (r, 0.5-r, 0.5)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data, labels; ratios=ratios, seed=1)

classes = sort(unique(yk))
n = c = length(classes)

##################################################
###                 Classifier                 ###
##################################################

# and encode labels to onehot
Xtrain = Xk
ytrain = yk
yoh_train = Flux.onehotbatch(ytrain, classes)
hdim = 64   # hidden dimension
ldim = 16   # latent dimension

# create a simple classificator model
mill_model = reflectinmodel(
    Xtrain,
    d -> Dense(d, hdim, swish),
    SegmentedMeanMax
)
model = Chain(
        mill_model, Mill.data,
        Dense(hdim, hdim, swish), Dense(hdim, hdim, swish),
        Dense(hdim, ldim), Dense(ldim, n)
)

# training parameters, loss etc.
opt = ADAM()
# loss(x, y) = Flux.logitcrossentropy(model(x), y)
loss(x, y) = Flux.logitcrossentropy(model(x), y)

accuracy(x, y) = round(mean(Flux.onecold(model(x), classes) .== y), digits=3)

using IterTools
using Flux: @epochs

batchsize = 64
function minibatch()
    ix = sample(1:nobs(Xk), batchsize)
    xb = reindex(Xk, ix)
    yb = yoh_train[:, ix]
    xb, yb
end

@epochs 1000 begin
    batches = map(_ -> minibatch(), 1:10)
    Flux.train!(loss, Flux.params(model), batches, opt)
    @show accuracy(Xt, yt)
end

# accuracy
accuracy(Xk, yk)    # known labels
accuracy(Xu, yu)    # unknown labels - not used for now
accuracy(Xt, yt)    # test data - this is the reference accuracy

# look at the created latent space

using Plots
ENV["GKSwstype"] = "100"
gr(markerstrokewidth=0, color=:jet, label="");

using master_thesis: encode
scatter2(model[1:end-1](Xk), zcolor=encode(yk, classes), ms=6)
scatter2!(model[1:end-1](Xu), zcolor=encode(yu, classes), opacity=0.5, marker=:square, markersize=2)
scatter2!(model[1:end-1](Xt), zcolor=encode(yt, classes), marker=:star)

# try kNN
using master_thesis: dist_knn
enc = model[1:end-1](Xk)
enc_u = model[1:end-1](Xu)
enc_t = model[1:end-1](Xt)
DMu = pairwise(Euclidean(), enc_u, enc)
foreach(k -> dist_knn(k, DMu, yk, yu), 1:4)

DMt = pairwise(Euclidean(), enc_t, enc)
foreach(k -> dist_knn(k, DMt, yk, yt), 1:4)

##############################################################################
###                 Classifier with Triplet regularization                 ###
##############################################################################

using ClusterLosses, Distances

# and encode labels to onehot
Xtrain = Xk
ytrain = yk
yoh_train = Flux.onehotbatch(ytrain, classes)
hdim = 16   # hidden dimension
ldim = 8    # latent dimension

# create a simple classificator model
mill_model_triplet = reflectinmodel(
    Xtrain,
    d -> Dense(d, hdim, swish),
    SegmentedMeanMax
)
model_triplet = Chain(
        mill_model_triplet, Mill.data,
        Dense(hdim, hdim, swish), Dense(hdim, hdim, swish),
        Dense(hdim, ldim), Dense(ldim, n)
)

# training parameters, loss etc.
margin = 1f0
α = 0.5f0

function loss_reg(x, yoh, y)
    ce = Flux.logitcrossentropy(model_triplet(x), yoh)
    enc = model_triplet[1:end-1](x)
    trl = ClusterLosses.loss(Triplet(margin), SqEuclidean(), enc, y)

    return ce + α*trl
end

accuracy(x, y) = round(mean(Flux.onecold(model_triplet(x), classes) .== y), digits=3)
opt = ADAM()

using IterTools
using Flux: @epochs

function minibatch(;batchsize=64)
    ix = sample(1:nobs(Xk), batchsize)
    xb = reindex(Xk, ix)
    yb = yoh_train[:, ix]
    yl = ytrain[ix]
    xb, yb, yl
end

@epochs 100 begin
    batches = map(_ -> minibatch(), 1:10)
    Flux.train!(loss_reg, Flux.params(model_triplet), batches, opt)
    @show loss_reg(batches[1]...)
    @show accuracy(Xt, yt)
end

# accuracy
accuracy(Xk, yk)    # known labels
accuracy(Xu, yu)    # unknown labels - not used for now
accuracy(Xt, yt)    # test data - this is the reference accuracy

# look at the created latent space
scatter2(model_triplet[1:end-1](Xk), zcolor=encode(yk, classes), ms=7)
scatter2!(model_triplet[1:end-1](Xu), zcolor=encode(yu, classes), opacity=0.5, marker=:square, markersize=2)
scatter2!(model_triplet[1:end-1](Xt), zcolor=encode(yt, classes), marker=:star)

using master_thesis: dist_knn
# try kNN
enc = model_triplet[1:end-1](Xk)
enc_u = model_triplet[1:end-1](Xu)
enc_t = model_triplet[1:end-1](Xt)
DMu = pairwise(Euclidean(), enc_u, enc)
foreach(k -> dist_knn(k, DMu, yk, yu), 1:5)

DMt = pairwise(Euclidean(), enc_t, enc)
foreach(k -> dist_knn(k, DMt, yk, yt), 1:5)

enc1 = vcat(enc, ones(1, size(enc, 2)))
enc_u1 = vcat(enc_u, ones(1, size(enc_u, 2)))
enc_t1 = vcat(enc_t, ones(1, size(enc_t, 2)))

emb = m.embedding
emb_u = transform(m, enc_u1)
emb_t = transform(m, enc_t1)

scatter2(emb, zcolor=encode(yk, classes))
scatter2!(emb_u, zcolor=encode(yu, classes), marker=:square)

DMu = pairwise(Euclidean(), emb_u, emb)
foreach(k -> dist_knn(k, DMu, yk, yu), 1:10)

DMt = pairwise(Euclidean(), emb_t, emb)
foreach(k -> dist_knn(k, DMt, yk, yt), 1:10)

using UMAP
enc = model[1:end-1](Xk)
enc_ = vcat(enc, ones(1, size(enc, 2)))
m = UMAP_(enc_, 2, n_neighbors=5)
enc_test = model[1:end-1](Xt)
enc_test_ = vcat(enc_test, ones(1, size(enc_test, 2)))
emb_test = transform(m, enc_test_)

scatter2(emb_test, zcolor=encode(yt, classes))

using Clustering
c = kmeans(pairwise(Euclidean(), emb_test), 3)
randindex(c, encode(yt, classes))
