using DrWatson
using master_thesis
using Distributions, DistributionsAD
using ConditionalDists
using Flux

using StatsBase
using LinearAlgebra
using Distances
using Base.Iterators: repeated

using Plots
gr(markerstrokewidth=0, color=:jet, label="");

using Mill

using master_thesis: reindex, seqids2bags
include(srcdir("point_cloud.jl"))

data = load_mnist_point_cloud()
ratios = (0.01, 0.49, 0.5)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.bag_labels; ratios=ratios)
length(yk)
countmap(yk)
lb = sort(unique(yk))
n = c = length(lb)

##################################################
###                 Classifier                 ###
##################################################

# and encode labels to onehot
Xtrain = Xk
ytrain = yk
yoh_train = Flux.onehotbatch(ytrain, lb)
hdim = 32

# create a simple classificator model
mill_model = reflectinmodel(
    Xtrain,
    d -> Dense(d, hdim, swish),
    SegmentedMeanMax
)
model = Chain(
        mill_model, Mill.data,
        Dense(hdim, hdim, swish), Dense(hdim, hdim, swish),
        Dense(hdim, 2), Dense(2, n)
)

# training parameters, loss etc.
opt = ADAM()
loss(x, y) = Flux.logitcrossentropy(model(x), y)
accuracy(x, y) = round(mean(lb[Flux.onecold(model(x))] .== y), digits=3)

using IterTools
using Flux: @epochs

function minibatch(;batchsize=64)
    ix = sample(1:nobs(Xk), batchsize)
    xb = reindex(Xk, ix)
    yb = yoh_train[:, ix]
    xb, yb
end

@epochs 100 begin
    for i in 1:30
        batch = minibatch()
        Flux.train!(loss, Flux.params(model), repeated(batch, 2), opt)
        # @show loss(batch...)
    end
    @show accuracy(Xtrain, ytrain)
    #ix = sample(1:nobs(Xt), 1000)
    #@show accuracy(reindex(Xt, ix), yt[ix])
end

# look at the created latent space
scatter2(model[1:end-1](Xk), zcolor=yk)
scatter2(model[1:end-1](Xu), zcolor=yu, opacity=0.5, marker=:square, markersize=2)
scatter2(model[1:end-1](Xt), zcolor=yt, marker=:star)

scatter2(model[1:end-1](Xu), zcolor=yu, marker=:square, markersize=2)

r = softmax(model[end](Xk))
i = 0
i += 1, heatmap(xh, yh, (x, y) -> softmax(model[end](vcat(x, y)))[i])

accuracy(Xk, yk)
accuracy(Xt, yt)


#######################################################
###                 Semi-supervised                 ###
#######################################################

include(scriptsdir("conditional_losses.jl"))
include(scriptsdir("conditional_bag_losses.jl"))

# parameters
c = n
dz = 2
xdim = 3

# mill model to get one-vector bag representation
bagmodel = Chain(reflectinmodel(
    Xk,
    d -> Dense(d, hdim),
    SegmentedMeanMax
), Mill.data, Dense(hdim, hdim, swish), Dense(hdim, 2));

# latent prior - isotropic gaussian
pz = MvNormal(zeros(Float32, zdim), 1f0)
# categorical prior
α = softmax(Float32.(randn(c)))

# categorical approximate
α_qy_x = Chain(Dense(2,2,swish), Dense(2,c),softmax)
qy_x = ConditionalCategorical(α_qy_x)

# encoder
net_xz = Chain(Dense(xdim+c,4,swish), Dense(4, 4, swish), SplitLayer(4, [zdim,zdim], [identity, safe_softplus]))
qz_xy = ConditionalMvNormal(net_xz)

# decoder
net_zx = Chain(Dense(zdim+c,4,swish), Dense(4, 4, swish), SplitLayer(4, [xdim,xdim], [identity, safe_softplus]))
px_yz = ConditionalMvNormal(net_zx)

# parameters and opt
ps = Flux.params(α, qz_xy, qy_x, px_yz, bagmodel)
opt = ADAM()


function minibatch(Xk, y, Xu;ksize=64, usize=64)
    kix = sample(1:nobs(Xk), ksize)
    uix = sample(1:nobs(Xu), usize)

    ye = encode(y, classes)
    xk, yk = [Xk[i] for i in kix], ye[kix]
    xu = [Xu[i] for i in uix]

    return xk, yk, xu
end

function accuracy(X, y, classes)
    N = length(y)
    ye = encode(y, classes)
    ynew = Flux.onecold(probs(condition(qy_x, bagmodel(X))), 1:length(classes))
    sum(ynew .== ye)/N
end
accuracy(X, y) = accuracy(X, y, classes)


function semisupervised_loss(xk, y, xu, N)
    # known and unknown losses
    l_known = loss_known_bag(xk, y)
    l_unknown = loss_unknown(xu)

    # classification loss on known data
    lc = 0.1 * N * loss_classification(xk, y)

    return l_known + l_unknown + lc
end

classes = sort(unique(yk))
ksize, usize = 32, 32
loss(xk, yk, xu) = semisupervised_loss(xk, yk, xu, ksize)
tr_batch = minibatch(Xk, yk, Xu; ksize=ksize, usize=usize);

for i in 1:30
    b = minibatch(Xk, yk, Xu; ksize=ksize, usize=usize);
    Flux.train!(loss, ps, zip((b...)), opt)
    # @show i
    # atr = round(accuracy(Xk, yk), digits=4)
    # ats = round(accuracy(Xt, yt), digits=4)
    # @info "Train accuracy: $atr"
    # @info "Test accuracy: $ats"
    @show mean(loss.(tr_batch[1], tr_batch[2], tr_batch[3]))
    @show accuracy(Xk, yk)
end

scatter2(project_data(Xk[30]), aspect_ratio=:equal, ms=(project_data(Xk[3])[3, :] .+ 3) .* 2)

function reconstruct(x, y)
    yoh = Float32.(Flux.onehotbatch(y, 1:n))
    xy = vcat(x, yoh)
    qz = condition(qz_xy, xy)
    z = rand(qz)
    yz = vcat(z, yoh)
    rand(condition(px_yz, yz))
end