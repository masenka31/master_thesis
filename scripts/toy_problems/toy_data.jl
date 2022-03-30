using DrWatson
@quickactivate
using Plots, Distributions
using master_thesis

# data
ndat = 500
X = rand(MvNormal([3,5],1),ndat)
x = X[1,:]
y = X[2,:]

scatter(x,y,aspect_ratio=:equal)

rmat(ϕ) = [cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)]

function moon(n, center, ϕ; sigma = 0.6, sigma2 = 1, radius = 2)
    R = rmat(ϕ)
    noise = rand(n) .* sigma
    theta = pi * rand(n)
    semi_up = hcat((radius .+ noise) .* cos.(theta) .+ center[1], (radius .+ noise) .* sin.(theta) .+ center[2])
    return collect((semi_up * R)') .+ randn(2, n) .* sigma2
end

n1 = 800
X1 = moon(n1, [0,0], pi/3, sigma=1, sigma2=0.7, radius=5)

n2 = 700
X2 = moon(n2, [3,3.2], 2.8/2*pi, sigma=1, sigma2=0.8, radius=5)

n3 = 400
X3 = rand(MvNormal([1, -2.5], [2.1 -0.4; -0.4 0.4]), n3)

y = vcat(
    ones(Int, n1),
    ones(Int, n2) .+ 1,
    ones(Int, n3) .+ 2
)

X = hcat(X1, X2, X3)
scatter2(X, color=y, aspect_ratio=:equal)


function split_semisupervised_data(X, y; ratios=(0.1,0.4,0.5), seed=nothing)
    
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    n = length(y)
    ix = sample(1:n, n, replace=false)
    nk, nu, nt = round.(Int, n .* ratios)

    Xk, Xu, Xt = X[:, ix[1:nk]], X[:, ix[nk+1:nk+nu]], X[:, ix[nk+nu+1:n]]
    yk, yu, yt = y[ix[1:nk]], y[ix[nk+1:nk+nu]], y[ix[nk+nu+1:n]]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return Xk, yk, Xu, yu, Xt, yt
end

Xk, yk, Xu, yu, Xt, yt = split_semisupervised_data(X, y, ratios=(0.01, 0.49, 0.5))
while length(unique(yk)) < 3
    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_data(X, y, ratios=(0.01, 0.49, 0.5))
end
scatter2(Xk, color=yk)
scatter2!(Xu, color=4, opacity=0.5)
scatter2!(Xt, color=:red, opacity=0.3, marker=:square, aspect_ratio=:equal, ms=2)

using DistributionsAD
using ConditionalDists
using Flux

using StatsBase
using LinearAlgebra
using Distances
using Base.Iterators: repeated

using Plots
gr(label="");

#######################################################
###                 Semi-supervised                 ###
#######################################################

# new conditional distributions
# number of classes can be fixed to be a global constant
const c = 3       # number of classes
zdim = 2          # latent dimension
xdim = 2          # input dimension
activation = swish
hdim = 5

net_xz = Chain(Dense(xdim+c,hdim,activation), Dense(hdim, hdim, activation), SplitLayer(hdim, [zdim,zdim], [identity, safe_softplus]))
net_zx = Chain(Dense(xdim+c,hdim,activation), Dense(hdim, hdim, activation), SplitLayer(hdim, [xdim,xdim], [identity, safe_softplus]))

α = softmax(Float32.(randn(c)))             # α is a trainable parameter!
qz_xy = ConditionalMvNormal(net_xz)
pz = MvNormal(zeros(Float32, zdim), 1f0)    # check that eltype is Float32

α_qy_x = Chain(Dense(2,hdim,activation), Dense(hdim,c),softmax)
qy_x = ConditionalCategorical(α_qy_x)

px_yz = ConditionalMvNormal(net_zx)

ps = Flux.params(α, qz_xy, qy_x, px_yz)
opt = ADAM(0.005)

# start testing loss functions
include(scriptsdir("conditional_losses.jl"))


function semisupervised_loss(xk, y, xu, N)
    # known and unknown losses
    # l_known = loss_known(xk, y)
    # l_unknown = loss_unknown(xu)

    # classification loss on known data
    lc = 0.1 * N * loss_classification(xk, y)

    # return l_known + l_unknown + lc
    return lc
end

function accuracy(X, y, c)
    N = length(y)
    ynew = map(xi -> Flux.onecold(condition(qy_x, xi).p, 1:c), eachcol(X))
    sum(ynew .== y)/N
end
accuracy(X, y) = accuracy(X, y, c)


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
@epochs 500 begin
    for i in 1:100
        batch = minibatch(Xk, yk, Xu; ksize=ksize, usize=usize)
        Flux.train!(loss, ps, repeated(batch, 1), opt)
    end
    atr = round(accuracy(Xk, yk), digits=4)
    ats = round(accuracy(Xt, yt), digits=4)
    println("Train known accuracy: $atr.")
    println("Test accuracy: $ats.")
end

function reconstruct(x, y)
    yoh = Flux.onehotbatch(y, 1:c)
    xy = vcat(x, yoh)

    # get the conditional and sample latent
    qz = condition(qz_xy, xy)
    z = rand(qz) # no need to do the reparametrization trick thanks to DistributionsAD
    yz = vcat(z, yoh)

    xnew = rand(condition(px_yz, yz))
end

function latent(x, y)
    yoh = Flux.onehotbatch(y, 1:c)
    xy = vcat(x, yoh)

    # get the conditional and sample latent
    qz = condition(qz_xy, xy)
    z = rand(qz) # no need to do the reparametrization trick thanks to DistributionsAD
end

function sample_new(y)
    z = randn(2, length(y))
    yoh = Flux.onehotbatch(y, 1:c)

    yz = vcat(z, yoh)
    xnew = rand(condition(px_yz, yz))
end

scatter2(sample_new(repeat([1], 1000)));
scatter2!(sample_new(repeat([2], 1000)));
scatter2!(sample_new(repeat([3], 1000)))

scatter2!(Xt, color=4, marker=:square, ms=2, markerstrokewidth=0)

##################################################
###                 Classifier                 ###
##################################################

classifier = Chain(Dense(xdim, hdim, activation), Dense(hdim, c))
classifier = Chain(Dense(xdim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, c))

const un = sort(unique(yk))
yoh = Flux.onehotbatch(yk, un)
data = Flux.Data.DataLoader((Xk, yoh))

classifier_accuracy(x, y) = sum(un[Flux.onecold(classifier(x))] .== y) ./ length(y)

classification_loss(x, y) = Flux.logitcrossentropy(classifier(x), y)

copt = ADAM()
@epochs 100 begin
    for i in 1:50 Flux.train!(classification_loss, Flux.params(classifier), data, copt) end
    @show classifier_accuracy(Xk, yk)
    @show classifier_accuracy(Xt, yt)
end