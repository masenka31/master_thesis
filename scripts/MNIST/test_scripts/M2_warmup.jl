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
ENV["GKSwstype"] = "100"
gr(markerstrokewidth=0, color=:jet, label="");

using Mill

using master_thesis: reindex, seqids2bags
using master_thesis: encode
using master_thesis.Models

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
    bdim = sample([2,4,8,16])           # the dimension of output of the HMill model
    batchsize = sample([64, 128, 256])
    agg = sample(["SegmentedMean", "SegmentedMax", "SegmentedMeanMax"])   # HMill aggregation function
    activation = sample(["swish", "relu", "tanh"])                  # activation function
    type = sample([:vanilla, :dense, :simple])
    # α = sample([0.1f0, 0.05f0, 0.01f0])
    α = sample([0.1f0, 1f0, 10f0])
    return parameters = (hdim = hdim, zdim = zdim, bdim = bdim, batchsize = batchsize, aggregation = agg, activation = activation, type = type, α = α)
end

parameters = sample_params()
model = M2_bag_constructor(Xk, c; parameters...)

function accuracy(model::M2BagModel, X, y, classes)
    ynew = Flux.onecold(condition(model.qy_x, model.bagmodel(X)).α, classes)
    mean(ynew .== y)
end
accuracy(X, y) = accuracy(model, X, y, classes)
@show accuracy(Xk, yk)

ye = encode(yk, classes)
batchsize = parameters.batchsize
function minibatch()
    kix = sample(1:nobs(Xk), batchsize)
    uix = sample(1:nobs(Xu), batchsize)

    xk, y = reindex(Xk, kix), ye[kix]
    xu = reindex(Xu, uix)
    return xk, y, xu
end

lknown(xk, y) = master_thesis.Models.loss_known_bag_Chamfer(model, xk, y, c)
lunknown(xu) = master_thesis.Models.loss_unknown_Chamfer(model, xu, c)

# reconstruction loss - known + unknown
function loss_rec(Xk, yk, Xu)
    l_known = mean(lknown.(Xk, yk))
    l_unknown = mean(lunknown.(Xu))
    return l_known + l_unknown
end
loss_rec_known(Xk, yk) = mean(lknown.(Xk, yk))

N = size(project_data(Xk), 2)
lclass(x, y) = loss_classification_crossentropy(model, x, y, c) * parameters.α * N

# now we should be able to dispatch over bags and labels
function lossf(Xk, yk, Xu)
    nk = nobs(Xk)
    bk = Flux.Zygote.@ignore [Xk[i] for i in 1:nk]

    nu = nobs(Xu)
    bu = Flux.Zygote.@ignore [Xu[i] for i in 1:nu]
    
    lr = loss_rec(bk, yk, bu)
    lc = lclass(Xk, yk)
    return lr + lc
end
@show lossf(minibatch()...)

function loss_warmup(Xk, yk, Xu)
    nk = nobs(Xk)
    bk = Flux.Zygote.@ignore [Xk[i] for i in 1:nk]
    lknown = loss_rec_known(bk, yk)
    ce = lclass(Xk, yk)
end
@show loss_warmup(minibatch()...)

# optimizer and training parameters
opt = ADAM(0.001)
ps = Flux.params(model)
max_accuracy = 0
best_model = deepcopy(model)
max_train_time = 60*3
batch = minibatch()

start_time = time()
while time() - start_time < max_train_time

    b = map(i -> minibatch(), 1:1)
    Flux.train!(loss_warmup, ps, b, opt)
    Flux.train!(lossf, ps, b, opt)
    @show loss_warmup(batch...)
    @show lossf(batch...)

    # @show accuracy(Xt, yt)
    @show a = accuracy(Xk, yk)
    @show accuracy(Xval, yval)
    if a >= max_accuracy
        max_accuracy = a
        best_model = deepcopy(model)
    end
end
plot12()

predict_label(X) = Flux.onecold(condition(model.qy_x, model.bagmodel(X)).α, classes)
cm, df = confusion_matrix(classes, Xk, yk, predict_label)
cm, df = confusion_matrix(classes, Xval, yval, predict_label)
cm, df = confusion_matrix(classes, Xt, yt, predict_label)

function reconstruct(model::M2BagModel, Xb, y, c)
    _, enc, yoh = master_thesis.Models.encoder(model, Xb, y, c)
    qz = condition(model.qz_xy, enc)
    z = rand(qz)
    yz = vcat(z, yoh)
    rand(condition(model.px_yz, yz))
end
reconstruct(Xb, y) = reconstruct(model, Xb, y, c)

function plot12(model=model)
    rec(Xb, y) = reconstruct(model, Xb, y, c)
    plt = []
    for i in 1:12
        i = sample(1:nobs(Xk))
        # i = 8
        Xb, y = Xk[i], ye[i]
        # Xb, y = Xk[i], 3
        Xhat = rec(Xb, y)
        p = scatter2(project_data(Xb), color=:3, xlims=(-3, 3), ylims=(-3, 3), axis=([], false), aspect_ratio=:equal, size=(400, 400), ms=project_data(Xb)[3, :] .+ 3 .* 1.5)
        p = scatter2!(Xhat, ms = Xhat[3, :] .+ 3 .* 1.5, opacity=0.7)
        push!(plt, p)
    end
    p = plot(plt..., layout=(3,4), size=(800,600))
    savefig("plot.png")
end