using DrWatson
@quickactivate
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
project_data(X::AbstractBagNode) = Mill.data(Mill.data(X))

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
hdim = 16

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
    for i in 1:10
        batch = minibatch()
        Flux.train!(loss, Flux.params(model), repeated(batch, 2), opt)
        @show loss(batch...)
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

accuracy(Xk, yk)
accuracy(Xt, yt)


#######################################################
###                 Semi-supervised                 ###
#######################################################

include(scriptsdir("conditional_losses.jl"))
include(scriptsdir("conditional_bag_losses.jl"))
include(scriptsdir("loss_functions.jl"))

# parameters
c = 10
zdim = 10
xdim = 3
hdim = 32
bdim = 2

# mill model to get one-vector bag representation
bagmodel = Chain(reflectinmodel(
    Xk,
    d -> Dense(d, hdim),
    SegmentedMeanMax
), Mill.data, Dense(hdim, hdim, swish), Dense(hdim, bdim));

# latent prior - isotropic gaussian
pz = MvNormal(zeros(Float32, zdim), 1f0)
# categorical prior
α = softmax(Float32.(ones(c)))

# categorical approximate
α_qy_x = Chain(Dense(bdim,hdim,swish), Dense(hdim,c),softmax)
qy_x = ConditionalCategorical(α_qy_x)

# encoder
net_xz = Chain(Dense(xdim+bdim+c,hdim,swish), Dense(hdim, hdim, swish), SplitLayer(hdim, [zdim,zdim], [identity, safe_softplus]))
qz_xy = ConditionalMvNormal(net_xz)

# decoder
net_zx = Chain(Dense(zdim+c,hdim,swish), Dense(hdim, hdim, swish), SplitLayer(hdim, [xdim,xdim], [identity, safe_softplus]))
px_yz = ConditionalMvNormal(net_zx)

# parameters and opt
ps = Flux.params(α, qz_xy, qy_x, px_yz, bagmodel)
opt = ADAM()

using master_thesis: encode
function minibatch(Xk, y, Xu;ksize=64, usize=64)
    kix = sample(1:nobs(Xk), ksize)
    uix = sample(1:nobs(Xu), usize)

    ye = encode(y, classes)
    xk, yk = [Xk[i] for i in kix], ye[kix]
    xu = [Xu[i] for i in uix]

    return xk, yk, xu
end
function minibatch_uniform(Xk, y, Xu;ksize=64, usize=64)
    k = round(Int, ksize / c)
    n = nobs(Xk)

    ik = []
    for i in 1:c
        avail_ix = (1:n)[y .== classes[i]]
        ix = sample(avail_ix, k)
        push!(ik, ix)
    end
    kix = shuffle(vcat(ik...))

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
    lc = 0.1f0 * N * loss_classification(xk, y)

    return l_known + l_unknown + lc
end
function semisupervised_loss(xk, y, xu, N)
    # known and unknown losses
    l_known = loss_known_bag(xk, y)
    l_unknown = loss_unknown(xu)

    # classification loss on known data
    n = size(project_data(xk), 2)
    # lc = 0.1 * N * n * loss_classification(xk, y)
    lc = 0.1 * N * loss_classification_crossentropy(xk, y)

    return l_known + l_unknown + lc
end
function semisupervised_warmup(xk, y, N)
    # known and unknown losses
    l_known = loss_known_bag(xk, y)

    # classification loss on known data
    n = size(project_data(xk), 2)
    lc = 0.1 * N * n * loss_classification(xk, y)

    return l_known + lc
end


classes = sort(unique(yk))
ksize, usize = 130, 130
loss(xk, yk, xu) = semisupervised_loss(xk, yk, xu, ksize)
loss(xk, yk, xu) = semisupervised_loss(xk, yk, xu, 700)
semisupervised_warmup(xk, y) = semisupervised_warmup(xk, y, 700)
tr_batch = minibatch_uniform(Xk, yk, Xu; ksize=ksize, usize=usize);

# warmup phase
for k in 1:50
    b = minibatch_uniform(Xk, yk, Xu; ksize=ksize, usize=usize);
    Flux.train!(semisupervised_warmup, ps, zip(b[1], b[2]), opt)
    @show accuracy(Xk, yk)
end

best_acc = 0
best_model = deepcopy.([α, qz_xy, qy_x, px_yz, bagmodel])

anim = @animate for i in 1:100
    plot_heatmap()
    # b = minibatch(Xk, yk, Xu; ksize=ksize, usize=usize);
    b = minibatch_uniform(Xk, yk, Xu; ksize=ksize, usize=usize);
    Flux.train!(loss, ps, zip((b...)), opt)
    # @show i
    # atr = round(accuracy(Xk, yk), digits=4)
    # ats = round(accuracy(Xt, yt), digits=4)
    # @info "Train accuracy: $atr"
    # @info "Test accuracy: $ats"
    @info "Epoch $i"
    @show mean(loss.(tr_batch[1], tr_batch[2], tr_batch[3]))
    a = accuracy(Xk, yk)
    @info "Accuracy = $a."
    if a > best_acc
        global best_acc = a
        global best_model = deepcopy.([α, qz_xy, qy_x, px_yz, bagmodel])
    end
end
gif(anim, "animation_crossentropy.gif", fps = 10)

# probabilities
r = probs(condition(qy_x, bagmodel(Xk)))
i = 0
i += 1;bar(r[i,1:100], color=Int.(yk[1:100]), size=(1000,400), ylims=(0,1), label="$(i-1)")

r = probs(condition(qy_x, bagmodel(Xu)))
i = 0
i += 1;bar(r[i,1:100], color=Int.(yu[1:100]), size=(1000,400), ylims=(0,1))

# latent space
latent = bagmodel(Xk)
scatter2(latent, zcolor=yk, ms=3)

latent = bagmodel(Xu)
scatter2(latent, zcolor=yu, ms=2)



# heatmap latent space
i = 0
xh = -5:0.1:10
yh = -10:0.1:7
i += 1; heatmap(xh, yh, (x, y) -> probs(condition(qy_x, vcat(x, y)))[i])

p = plot(layout=(2,5), legend=false, axis=([], false));
for i in 1:10
    p = heatmap!(
        xh, yh, (x, y) -> probs(condition(qy_x, vcat(x, y)))[i], subplot=i,
        legend=:none, title="no. $(i-1)", titlefontsize=7, size=(1000, 600)
    )
end
p
wsave(plotsdir("heatmap100epochs.svg"), p)

function plot_heatmap()
    p = plot(layout=(2,5), legend=false, axis=([], false));
    for i in 1:10
        p = heatmap!(
            xh, yh, (x, y) -> probs(condition(qy_x, vcat(x, y)))[i], subplot=i,
            legend=:none, title="no. $(i-1)", titlefontsize=7, size=(1000, 600)
        )
    end
    p
end

function reconstruct(Xb, y)
    Xdata = project_data(Xb)
    n = size(Xdata, 2)
    ydata = repeat([y], n)

    v = bagmodel(Xb)
    vdata = reshape(repeat(v, n), length(v), n)

    # get the concatenation with label (onehot encoding?)
    yoh = Flux.onehotbatch(ydata, 1:c)
    xy = vcat(Xdata, yoh, vdata)

    # get the conditional and sample latent
    qz = condition(qz_xy, xy)
    z = mean(qz)
    yz = vcat(z, yoh)
    rand(condition(px_yz, yz))
end

plot_number(i) = scatter2(project_data(Xk[i]), aspect_ratio=:equal, ms=(project_data(Xk[i])[3, :] .+ 3) .* 2, size=(400, 400), marker=:square)

i = 0
i += 1; scatter2(project_data(Xk[i]), aspect_ratio=:equal, ms=(project_data(Xk[i])[3, :] .+ 3) .* 2, size=(400, 400), marker=:square, label=yk[i])

i+=1; plot_number(i); scatter2!(reconstruct(Xk[i], yk[i] + 1), ms=reconstruct(Xk[i], yk[i] + 1)[3, :] .+ 3 .* 2, color=:green)

pvec = []
k = sample(1:600)
for i in k:k+8
    p_i = plot_number(i)

    R = reconstruct(Xk[i], yk[i] + 1)
    r = classes[Flux.onecold(condition(qy_x, bagmodel(Xk[i])).α)][1]
    p_i = scatter2!(
        R, ms=R[3, :] .+ 3 .* 2,
        color=:green, axis=([], false), xlims=(-3,3), ylims=(-3, 3), size=(900, 900),
        title="predicted: $r", titlefontsize=8
    )
    push!(pvec, p_i)
end
plot(pvec..., layout=(3,3), suze=(900,900))