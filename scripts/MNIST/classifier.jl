using DrWatson
@quickactivate
using master_thesis
using Distributions, DistributionsAD
using ConditionalDists
using Flux

using StatsBase, Random
using LinearAlgebra
using Distances
using Base.Iterators: repeated

using Plots
gr(markerstrokewidth=0, color=:jet, label="");

using Mill

using master_thesis: reindex, seqids2bags
using master_thesis: encode
include(srcdir("point_cloud.jl"))
project_data(X::AbstractBagNode) = Mill.data(Mill.data(X))

# load MNIST data and split it
data = load_mnist_point_cloud()
b = map(x -> any(x .== [0,1,4]), data.bag_labels)
filt_data, filt_labels = reindex(data.data, b), data.bag_labels[b]
ratios = (0.01, 0.49, 0.5)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.bag_labels; ratios=ratios)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(filt_data, filt_labels; ratios=ratios)
length(yk)
countmap(yk)
classes = sort(unique(yk))
n = c = length(classes)

##################################################
###                 Classifier                 ###
##################################################

# and encode labels to onehot
Xtrain = Xk
ytrain = yk
yoh_train = Flux.onehotbatch(ytrain, classes)
hdim = 32   # hidden dimension
ldim = 8    # latent dimension

# create a simple classificator model
mill_model = reflectinmodel(
    Xtrain,
    d -> Dense(d, hdim, swish),
    SegmentedMeanMax
)
model = Chain(
        mill_model, Mill.data,
        Dense(hdim, hdim, swish), Dense(hdim, hdim, swish),
        Dense(hdim, ldim), Dense(ldim, n, sigmoid)
)

# training parameters, loss etc.
opt = ADAM()
# loss(x, y) = Flux.logitcrossentropy(model(x), y)
loss(x, y) = Flux.mae(model(x), y)

safe_log(x) = log(x + 10e-20)
function loss_open(X, yoh)
    yhat = model(X)
    l1 = - sum(safe_log.(yhat[yoh]))
    l2 = - sum(safe_log.(1 .- yhat[.!yoh]))
    return l1 + l2
end

accuracy(x, y) = round(mean(classes[Flux.onecold(model(x))] .== y), digits=3)

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
    # Flux.train!(loss, Flux.params(model), batches, opt)
    Flux.train!(loss_open, Flux.params(model), batches, opt)
    @show loss_open(batches[1]...)
    @show accuracy(Xk, yk)
end

# accuracy
accuracy(Xk, yk)    # known labels
accuracy(Xu, yu)    # unknown labels - not used for now
accuracy(Xt, yt)    # test data - this is the reference accuracy

# look at the created latent space
scatter2(model[1:end-1](Xk), zcolor=yk)
scatter2(model[1:end-1](Xu), zcolor=yu, opacity=0.5, marker=:square, markersize=2)
scatter2(model[1:end-1](Xt), zcolor=yt, marker=:star)

scatter2(model[1:end-1](Xu), zcolor=yu, marker=:square, markersize=2)

i = 0
xh = -40:0.5:70
yh = -60:0.5:50
i += 1; heatmap(xh, yh, (x, y) -> softmax(model[end](vcat(x, y)))[i])

p = plot(layout=(2,5), legend=false, axis=([], false), size=(1000, 600));
for i in 1:10
    p = heatmap!(xh, yh, (x, y) -> softmax(model[end](vcat(x, y)))[i],
    subplot=i, legend=:none, title="no. $i", titlefontsize=8)
end
p

enc = model(Xt)

# those that should be classified right
b = map(x -> sum(x .== 1) == 1, eachcol(enc))
b = map(x -> sum(x .> 0.99) == 1, eachcol(enc))

# those that have multiple classes possible
b = map(x -> sum(sum(x .> 0.9) > 1), eachcol(enc)) |> BitVector
sum(b)

# get the accuracy on this data
xx = reindex(Xt, (1:nobs(Xt))[b])
yy = yt[b]
accuracy(xx, yy)