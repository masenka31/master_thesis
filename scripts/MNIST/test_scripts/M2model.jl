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

# load MNIST data and split it
data = load_mnist_point_cloud()
b = map(x -> any(x .== [0,1,3,4]), data.bag_labels)
filt_data, filt_labels = reindex(data.data, b), data.bag_labels[b]
r = 0.002
ratios = (r, 0.5-r, 0.5)
seed = 1
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.bag_labels; ratios=ratios, seed=1)
# Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(filt_data, filt_labels; ratios=ratios)
length(yk)
countmap(yk)
classes = sort(unique(yk))
n = c = length(classes)

Xval, yval, Xu, yu = validation_data(yk, Xu, yu, seed, classes)

################################################################
###                 Semi-supervised M2 model                 ###
################################################################

function sample_params()
    hdim = sample([16,32,64])           # hidden dimension
    zdim = sample([2,4,8,16])           # latent dimension
    bdim = sample([2,4,8,16])           # the dimension ob output of the HMill model
    batchsize = sample([64, 128, 256])
    agg = sample(["SegmentedMean", "SegmentedMax", "SegmentedMeanMax"])   # HMill aggregation function
    activation = sample(["swish", "relu", "tanh"])                  # activation function
    type = sample([:vanilla, :dense, :simple])
    # α = sample([0.1f0, 0.05f0, 0.01f0])
    α = sample([1f0, 10f0, 100f0])
    return parameters = (hdim = hdim, zdim = zdim, bdim = bdim, batchsize = batchsize, aggregation = agg, activation = activation, type = type, α = α)
end

parameters = sample_params()
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
net_xz = Chain(Dense(xdim+c,hdim,swish), Dense(hdim, hdim, swish), SplitLayer(hdim, [zdim,zdim], [identity, safe_softplus]))
qz_xy = ConditionalMvNormal(net_xz)

# decoder
net_zx = Chain(Dense(zdim+c,hdim,swish), Dense(hdim, hdim, swish), SplitLayer(hdim, [xdim,xdim], [identity, safe_softplus]))
px_yz = ConditionalMvNormal(net_zx)

# parameters and opt
ps = Flux.params(α, qz_xy, qy_x, px_yz, bagmodel)
opt = ADAM()

function accuracy(X, y, classes)
    N = length(y)
    ye = encode(y, classes)
    ynew = Flux.onecold(probs(condition(qy_x, bagmodel(X))), 1:length(classes))
    sum(ynew .== ye)/N
end
accuracy(X, y) = accuracy(X, y, classes)

NN = mean(length.(Xk.bags))
function semisupervised_loss(xk, y, xu, N)
    # known and unknown losses
    # l_known = Float32(NN) * loss_known_bag(xk, y)
    l_known = loss_known_bag(xk, y)
    # l_unknown = loss_unknown_bag(xu)
    l_unknown = loss_unknown(xu)

    # classification loss on known data
    lc = N * loss_classification(xk, y)

    return l_known + l_unknown + lc
end

function minibatch_uniform()
    k = round(Int, ksize / c)
    n = nobs(Xk)

    ik = []
    for i in 1:c
        avail_ix = (1:n)[yk .== classes[i]]
        ix = sample(avail_ix, k)
        push!(ik, ix)
    end
    kix = shuffle(vcat(ik...))
    uix = sample(1:nobs(Xu), usize)

    # encode labels to 1:c
    ye = encode(yk, classes)

    xk, y = reindex(Xk, kix), ye[kix]
    xu = reindex(Xu, uix)
    return xk, y, xu
end

ksize, usize = 60,60
loss(xk, yk, xu) = semisupervised_loss(xk, yk, xu, ksize)
loss(xk, yk, xu) = semisupervised_loss(xk, yk, xu, nobs(Xk))
tr_batch = minibatch_uniform();

best_model = deepcopy.([α, qz_xy, qy_x, px_yz])
best_acc = 0

anim = @animate for i in 1:1000
    # plot_heatmap()
    # enc = bagmodel(Xk)
    # scatter3(enc, zcolor=yk, ms=4, xlims=(-10,10), ylims=(-10, 10), zlims=(-10, 10))

    # b = map(_ -> minibatch_uniform(), 1:1)
    b = minibatch_uniform()
    Flux.train!(loss, ps, zip(b...), opt)

    @info "Epoch $i"
    # @show loss(tr_batch...)
    @show mean(loss.(tr_batch...))
    a = accuracy(Xk, yk)
    @info "Accuracy = $a."

    if a >= best_acc
        @info "Accuracy improved."
        global best_acc = a
        global best_model = deepcopy.([α, qz_xy, qy_x, px_yz])
    end
end
gif(anim, "latent_space_previousM2.gif", fps = 20)

# probabilities
r = probs(condition(qy_x, bagmodel(Xk)))
i = 0
i += 1;bar(r[i,1:100], color=Int.(yk[1:100]), size=(1000,400), ylims=(0,1), label="$(classes[i])")

r = probs(condition(qy_x, bagmodel(Xu)))
i = 0
i += 1;bar(r[i,1:100], color=Int.(yu[1:100]), size=(1000,400), ylims=(0,1))

# latent space
enc = bagmodel(Xk)
scatter2(enc, zcolor=yk, ms=4)

enc = bagmodel(Xu)
scatter2(enc, zcolor=yu, ms=2)
enc = bagmodel(Xt)
scatter2(enc, zcolor=yt, ms=2)



# heatmap latent space
i = 0
xh = -10:0.1:10
yh = -15:0.1:15
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
function plot_heatmap3()
    p = plot(layout=(1,3), legend=false, axis=([], false));
    for i in 1:3
        p = heatmap!(
            xh, yh, (x, y) -> probs(condition(qy_x, vcat(x, y)))[i], subplot=i,
            legend=:none, title="no. $(classes[i])", titlefontsize=7, size=(1000, 600)
        )
    end
    p
end

function reconstruct(Xb, y)

    Xdata = project_data(Xb)
    n = size(Xdata, 2)
    ydata = repeat([y], n)

    # get the concatenation with label (onehot encoding?)
    yoh = Flux.onehotbatch(ydata, 1:c)
    xy = vcat(Xdata, yoh)

    # get the conditional and sample latent
    qz = condition(qz_xy, xy)
    z = rand(qz)
    yz = vcat(z, yoh)
    rand(condition(px_yz, yz))
end

plot_number(i) = scatter2(project_data(Xk[i]), aspect_ratio=:equal, ms=(project_data(Xk[i])[3, :] .+ 3) .* 2, size=(400, 400), marker=:square)

i = 0
i += 1; scatter2(project_data(Xk[i]), aspect_ratio=:equal, ms=(project_data(Xk[i])[3, :] .+ 3) .* 2, size=(400, 400), marker=:square, label=yk[i])

i+=1; plot_number(i); scatter2!(reconstruct(Xk[i], yk[i] + 1), ms=reconstruct(Xk[i], yk[i] + 1)[3, :] .+ 3 .* 2, color=:green)

k = sample(1:200)
pvec = []
for i in k:k+8
    p_i = plot_number(i)
    yi = encode([yk[i]], classes)[1]
    R = reconstruct(Xk[i], yi)
    r = classes[Flux.onecold(condition(qy_x, bagmodel(Xk[i])).α)][1]
    p_i = scatter2!(
        R, ms=R[3, :] .+ 3 .* 2,
        color=:green, axis=([], false), xlims=(-3,3), ylims=(-3, 3), size=(900, 900),
        title="predicted: $r\ntrue: $(yk[i])", titlefontsize=8
    )
    push!(pvec, p_i)
end
plot(pvec..., layout=(3,3), size=(900,900))

plot_number(i) = scatter2(project_data(Xt[i]), aspect_ratio=:equal, ms=(project_data(Xk[i])[3, :] .+ 3) .* 2, size=(400, 400), marker=:square)

k = sample(1:200)
pvec = []
for i in k:k+8
    # p_i = plot_number(i)
    yi = encode([yt[i]], classes)[1]
    R = reconstruct(Xt[i], yi)
    r = classes[Flux.onecold(condition(qy_x, bagmodel(Xt[i])).α)][1]
    p_i = scatter2(
        R, ms=R[3, :] .+ 3 .* 2,
        color=:green, axis=([], false), xlims=(-3,3), ylims=(-3, 3), size=(900, 900),
        title="predicted: $r\ntrue: $(yt[i])", titlefontsize=8
    )
    push!(pvec, p_i)
end
plot(pvec..., layout=(3,3), size=(900,900))