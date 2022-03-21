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
include(srcdir("toy", "moons.jl"))

# get moons array data
n = 10
data, y, lnum = generate_moons_data(n, 600; λ=50, gap=1, max_val = 10)
X = hcat(data...)
scatter2(X, zcolor=lnum,aspect_ratio=:equal, ms=2.5)

# choice of bags
ix = map(i -> findfirst(x -> x == i, y), 1:n)
scatter2(data[ix[1]], color=1)
for i in 2:10 scatter2!(data[i], color=i) end
plot!(aspect_ratio=:equal, size=(800,800))

# create mill data
mill_data = BagNode(ArrayNode(hcat(data...)), get_obs(data))
project_data(X::AbstractBagNode) = Mill.data(Mill.data(X))

##################################################
###                 Classifier                 ###
##################################################

# split known data to train/test
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_data(mill_data, y, ratios=(0.1, 0.4, 0.5))
while length(countmap(yk)) < n
    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_data(mill_data, y, ratios=(0.1, 0.4, 0.5))
end

# split balanced
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(mill_data, y, ratios=(0.1, 0.4, 0.5))

# and encode labels to onehot
Xtrain = Xk
ytrain = yk
yoh_train = Flux.onehotbatch(ytrain, 1:n)

# create a simple classificator model
mill_model = reflectinmodel(
    Xtrain,
    d -> Dense(d, 2)
)
model = Chain(mill_model, Mill.data, Dense(2, n))

# training parameters, loss etc.
opt = ADAM()
loss(x, y) = Flux.logitcrossentropy(model(x), y)
accuracy(x, y) = round(mean(collect(1:n)[Flux.onecold(model(x))] .== y), digits=3)

using IterTools
using Flux: @epochs

@epochs 1000 begin
    Flux.train!(loss, Flux.params(model), repeated((Xtrain, yoh_train), 5), opt)
    @show loss(Xtrain, yoh_train)
    @show accuracy(Xtrain, ytrain)
    @show accuracy(Xt, yt)
end

# look at the created latent space
scatter2(model[1:end-1](Xk), zcolor=yk)
scatter2!(model[1:end-1](Xu), zcolor=yu, opacity=0.5, marker=:square)
scatter2!(model[1:end-1](Xt), zcolor=yt, marker=:star)

#######################################################
###                 Semi-supervised                 ###
#######################################################

include(scriptsdir("conditional_losses.jl"))
include(scriptsdir("conditional_bag_losses.jl"))

# mill model to get one-vector bag representation
bagmodel = Chain(reflectinmodel(
    Xk,
    d -> Dense(d, 2),
    SegmentedMeanMax
), Mill.data)

# parameters
c = n       # number of classes
zdim = 2      # latent dimension
xdim = 2    # input dimension

# latent prior - isotropic gaussian
pz = MvNormal(zeros(Float32, zdim), 1f0)    # check that eltype is Float32
# categorical prior
α = softmax(Float32.(randn(c)))             # α is a trainable parameter!

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

# minibatch function for bags
function minibatch(Xk, y, Xu;ksize=64, usize=64)
    kix = sample(1:nobs(Xk), ksize)
    uix = sample(1:nobs(Xu), usize)

    xk, yk = Xk[kix], y[kix]
    xu = Xu[uix]

    return xk, yk, xu
end
function minibatch(Xk, y, Xu;ksize=64, usize=64)
    if ksize > length(y)
        xk, yk = [Xk[i] for i in 1:nobs(Xk)], y
    else
        kix = sample(1:nobs(Xk), ksize, replace=false)
        xk, yk = [Xk[i] for i in kix], y[kix]
    end

    if usize > nobs(Xu)
        xu = [Xu[i] for i in 1:nobs(Xu)]
    else    
        uix = sample(1:nobs(Xu), usize, replace=false)
        xu = [Xu[i] for i in uix]
    end

    return xk, yk, xu
end

function accuracy(X, y, c)
    N = length(y)
    ynew = Flux.onecold(probs(condition(qy_x, bagmodel(X))), 1:c)
    sum(ynew .== y)/N
end
accuracy(X, y) = accuracy(X, y, c)

function semisupervised_loss(xk, y, xu, N)
    # known and unknown losses
    l_known = loss_known_mill(xk, y)
    l_unknown = loss_unknown_mill(xu)

    # classification loss on known data
    lc = 0.1 * N * loss_classification_mill(xk, y)

    return l_known + l_unknown + lc
end
function semisupervised_loss(xk, y, xu, N)
    # known and unknown losses
    l_known = loss_known_bag(xk, y)
    l_unknown = loss_unknown(xu)

    # classification loss on known data
    lc = 0.1 * N * loss_classification(xk, y)

    return l_known + l_unknown + lc
end

ksize, usize = 64, 64
loss(xk, yk, xu) = semisupervised_loss(xk, yk, xu, ksize*1.5)
test_batch = minibatch(Xk, yk, Xu)

for i in 1:50
    for k in 1:2
        b = minibatch(Xk, yk, Xu; ksize=ksize, usize=usize);
        Flux.train!(loss, ps, zip((b...)), opt)
    end
    @show i
    atr = round(accuracy(Xk, yk), digits=4)
    ats = round(accuracy(Xt, yt), digits=4)
    @info "Train accuracy: $atr"
    @info "Test accuracy: $ats"
    @show loss(test_batch[1][1], test_batch[2][1], test_batch[3][1])
end

r = probs(condition(qy_x, bagmodel(Xk)))
i = 0
i += 1;bar(r[i,:], color=Int.(yk), size=(1000,400), ylims=(0,1))


latent = bagmodel(Xk)
scatter2(latent, zcolor=yk, ms=7)
latent_unknown = bagmodel(Xu)
scatter2!(latent_unknown, zcolor=yu, opacity=0.6, marker=:square, ms=2)
latent_test = bagmodel(Xt)
scatter2!(latent_test, zcolor=yt, marker=:star)

scatter2(reconstruct_mean(Xk[1], yk[1]), color=:blue)
scatter2!(reconstruct_mean(Xk[2], yk[2]), color=:green)
scatter2!(reconstruct_mean(Xk[3], yk[3]), color=:red)

scatter2!(project_data(Xk[1]), color=:blue, marker=:square)
scatter2!(project_data(Xk[2]), color=:green, marker=:square)
scatter2!(project_data(Xk[3]), color=:red, marker=:square)
plot!(aspect_ratio=:equal)

r = condition(qy_x, bagmodel(Xk)).α
bar(r[10, :], color=Int.(yk), ylims=(0,1))

scatter2(project_data(Xk[1]), color=:blue, marker=:square, opacity=0.5)
scatter2!(project_data(Xk[2]), color=:green, marker=:square, opacity=0.5)
scatter2!(project_data(Xk[3]), color=:red, marker=:square, opacity=0.5)

scatter2!(reconstruct_rand(Xk[1], yk[1]), color=:blue, ms=6)
scatter2!(reconstruct_rand(Xk[2], yk[2]), color=:green, ms=6)
scatter2!(reconstruct_rand(Xk[3], yk[3]), color=:red, ms=6)

plot!(aspect_ratio=:equal)
