# semisupervised VAE

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
gr(label="");

# new conditional distributions
c = 3       # number of classes
zdim = 2    # latent dimension
xdim = 2    # input dimension
activation = tanh
hdim = 2

net_xz = Chain(Dense(xdim+c,hdim,activation), Dense(hdim, hdim, activation), SplitLayer(hdim, [zdim,zdim], [identity, safe_softplus]))
net_zx = Chain(Dense(xdim+c,hdim,activation), Dense(hdim, hdim, activation), SplitLayer(hdim, [xdim,xdim], [identity, safe_softplus]))

α = softmax(Float32.(randn(c)))             # α is a trainable parameter!
qz_xy = ConditionalMvNormal(net_xz)
pz = MvNormal(zeros(Float32, zdim), 1f0)    # check that eltype is Float32

α_qy_x = Chain(Dense(2,2,activation), Dense(2,c),softmax)
qy_x = ConditionalCategorical(α_qy_x)

px_yz = ConditionalMvNormal(net_zx)

ps = Flux.params(α, qz_xy, qy_x, px_yz)
opt = ADAM(0.005)

# start testing loss functions
include(scriptsdir("conditional_losses.jl"))
x = randn(Float32, 2)
y = 2
xb = randn(Float32, 2, 10)
yb = sample(1:c, 10)
loss_known(x, y)
loss_known(xb, yb)
loss_unknown(x)
loss_unknown(xb)
loss_classification(x, y)
loss_classification(xb, yb)

function semisupervised_loss(xk, y, xu, N)
    # known and unknown losses
    l_known = loss_known(xk, y)
    l_unknown = loss_unknown(xu)

    # classification loss on known data
    lc = 0.1 * N * loss_classification(xk, y)

    return l_known + l_unknown + lc
end

# load data
include(srcdir("toy", "data.jl"))
ratios = (0.1, 0.4, 0.5)
n1, n2, n3 = 220, 170, 210
X, y = generate_data(n1, n2, n3)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_data(Float32.(X), y; ratios=ratios)
space_plot()

function accuracy(X, y, c)
    N = length(y)
    ynew = map(xi -> Flux.onecold(condition(qy_x, xi).p, 1:c), eachcol(X))
    sum(ynew .== y)/N
end
accuracy(X, y) = accuracy(X, y, c)

# this only works for Xk and Xu having the same length
# data = Flux.Data.DataLoader((Xk, yk, Xu), batchsize=64)

# create minibatch
function minibatch(Xk, y, Xu;ksize=64, usize=64)
    kix = sample(1:size(Xk, 2), ksize)
    uix = sample(1:size(Xu, 2), usize)

    xk, yk = Xk[:, kix], y[kix]
    xu = Xu[:, uix]

    return xk, yk, xu
end
data = minibatch(Xk, yk, Xu)

ksize, usize = 64, 64
loss(xk, y, xu) = semisupervised_loss(xk, y, xu, 64)

using Flux: @epochs
@epochs 2000 begin
    batch = minibatch(Xk, yk, Xu; ksize=ksize, usize=usize)
    Flux.train!(loss, ps, repeated(batch, 1), opt)
    a = round(accuracy(Xt, yt), digits=4)
    @show a
end

anim = @animate for i in 1:300
    space_plot()
    batch = minibatch(Xk, yk, Xu; ksize=ksize, usize=usize)
    Flux.train!(loss, ps, repeated(batch, 1), opt)
    @show i
    a = round(accuracy(Xt, yt), digits=4)
    @show a
end
gif(anim, "animation.gif", fps = 15)

space_plot()

r = mapreduce(xi -> probs(condition(qy_x, xi)), hcat, eachcol(Xt))
bar(r[1,:], color=Int.(yt), size=(1000,400))

function sample_new(y)
    yoh = Flux.onehotbatch(y, 1:c)
    z = rand(pz)
    yz = vcat(z, yoh)
    rand(condition(px_yz, yz))
end
function sample_new(y, n)
    yv = repeat([y], n)
    yoh = Flux.onehotbatch(yv, 1:c)
    z = rand(pz, n)
    yz = vcat(z, yoh)
    rand(condition(px_yz, yz))
end

function reconstruct(x, y)
    yoh = Flux.onehotbatch(y, 1:c)
    xy = vcat(x, yoh)
    qz = condition(qz_xy, xy)
    z = rand(qz)
    yz = vcat(z, yoh)
    rand(condition(px_yz, yz))
end

scatter2(sample_new(1, 100))
scatter2!(sample_new(2, 100))
scatter2!(sample_new(3, 100))
scatter2!(X)

scatter2(reconstruct(X, y), color=Int.(y));
scatter2!(X, color=4, opacity=0.6)