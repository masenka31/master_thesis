using DrWatson
@quickactivate
using master_thesis
using master_thesis.Models
using Distributions, DistributionsAD
using ConditionalDists
using Flux

using StatsBase, Random
using LinearAlgebra
using Distances
using Base.Iterators: repeated

using Mill

using master_thesis: reindex, seqids2bags, encode
using master_thesis.Models: chamfer_score

include(srcdir("point_cloud.jl"))

r = 0.002
ratios = (r, 0.5-r, 0.5)        # get the ratios
data = load_mnist_point_cloud()
seed = 1

b = map(x -> any(x .== [0,1,3,4]), data.bag_labels)
filt_data, filt_labels = reindex(data.data, b), data.bag_labels[b]
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(filt_data, filt_labels; ratios=ratios, seed=seed)

classes = sort(unique(yk))
n = c = length(classes)

struct ChamferKLModel
    prior
    bagmodel
    classifier
    encoder
    generator
    decoder
end

Flux.@functor ChamferKLModel

function Flux.trainable(m::ChamferKLModel)
    (bagmodel=m.bagmodel, classifier=m.classifier, encoder=m.encoder, generator=m.generator, decoder=m.decoder, )
end

function ChamferKLModel_constructor(Xk, c; cdim=2, hdim=4, bdim=4, gdim=4, aggregation=SegmentedMeanMax, activation=swish, kwargs...)
    # get activation function
    if typeof(activation) == String
        activation = eval(Symbol(activation))
    end

    # prior
    prior = MvNormal(zeros(Float32, cdim), 1f0)

    # HMill model
    bagmodel = Chain(reflectinmodel(
        Xk,
        d -> Dense(d, hdim),
        aggregation
    ), Mill.data, Dense(hdim, bdim))

    # classifier
    classifier = Chain(Dense(bdim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, c))

    # encoder
    net_encoder = Chain(Dense(bdim+c,hdim,activation), Dense(hdim, hdim, activation), SplitLayer(hdim, [cdim,cdim], [identity, safe_softplus]))
    encoder = ConditionalMvNormal(net_encoder)

    # generator
    net_generator = Chain(Dense(cdim+c,hdim,activation), Dense(hdim, hdim, activation), Dense(hdim, hdim, activation), SplitLayer(hdim, [gdim,gdim], [identity, safe_softplus]))
    # net_generator = Chain(Dense(cdim,hdim,activation), Dense(hdim, hdim, activation), Dense(hdim, hdim, activation), SplitLayer(hdim, [gdim,gdim], [identity, safe_softplus]))
    generator = ConditionalMvNormal(net_generator)

    # decoder
    xdim = size(Xk[1].data.data, 1)
    decoder = Chain(Dense(gdim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, xdim))

    return ChamferKLModel(prior, bagmodel, classifier, encoder, generator, decoder)
end

m = ChamferKLModel_constructor(Xk, c; cdim=8, hdim=32, bdim=8, gdim=8)

using Flux3D: chamfer_distance

# we can give onehot encoding from the beginning?
function loss_known(m::ChamferKLModel, Xb, y::Int, c)
    X = project_data(Xb)
    n = size(X, 2)
    x = m.bagmodel(Xb)
    yoh = Flux.onehot(y, 1:c)
    qc = condition(m.encoder, vcat(x, yoh))
    context = rand(qc)

    kl_loss = mean(kl_divergence(qc, m.prior))

    cy = vcat(context, yoh)
    Z = mapreduce(i -> rand(condition(m.generator, cy)), hcat, 1:n)

    Xhat = m.decoder(Z)

    return chamfer_distance(X, Xhat) + kl_loss
end

function loss_unknown(m::ChamferKLModel, Xb, c, loss_known)
    lmat = map(y -> loss_known(m, Xb, y, c), 1:c)
    prob = softmax(m.classifier(m.bagmodel(Xb)))
    l = sum(prob .* lmat)
    e = - entropy(prob)
    return l + e
end

loss_classification(m::ChamferKLModel, Xb, y, c) = Flux.logitcrossentropy(m.classifier(m.bagmodel(Xb)), Flux.onehotbatch(y, 1:c))

function accuracy(m::ChamferKLModel, X, y, classes)
    ynew = Flux.onecold(m.classifier(m.bagmodel(X)), classes)
    mean(ynew .== y)
end
accuracy(X, y) = accuracy(m, X, y, classes)

lknown(xk, y) = loss_known(m, xk, y, c)
lunknown(xu) = loss_unknown(m, xu, c, loss_known)

function loss_rec(Xk, yk, Xu)
    l_known = mean(lknown.(Xk, yk))
    l_unknown = mean(lunknown.(Xu))
    return l_known + l_unknown
end

N = nobs(Xk)
α = 0.1f0
lclass(x, y) = loss_classification(m, x, y, c) * α * N

function lossf(Xk, yk, Xu)
    nk = nobs(Xk)
    bk = Flux.Zygote.@ignore [Xk[i] for i in 1:nk]

    nu = nobs(Xu)
    bu = Flux.Zygote.@ignore [Xu[i] for i in 1:nu]
    
    lr = loss_rec(bk, yk, bu)
    lc = lclass(Xk, yk)
    return lr + lc
end

ye = encode(yk, classes)
batchsize = 64
function minibatch()
    kix = sample(1:nobs(Xk), batchsize)
    uix = sample(1:nobs(Xu), batchsize)

    xk, y = reindex(Xk, kix), ye[kix]
    xu = reindex(Xu, uix)
    return xk, y, xu
end

ps = Flux.params(m)
# opt = ADAM()
opt = ADAM(0.01)

batch = minibatch()
@show lossf(batch...)

using Flux: @epochs
@epochs 50 begin
    b = map(i -> minibatch(), 1:5)
    Flux.train!(lossf, ps, b, opt)
    @show lossf(batch...)
    @show accuracy(Xk, yk)
    @show accuracy(Xu, yu)
end

using Plots
ENV["GKSwstype"] = "100"
gr(markerstrokewidth=0, color=:jet, label="");

scatter3(m.bagmodel(Xk), zcolor=yk, ms=4)
scatter3!(m.bagmodel(Xu), zcolor=yu, opacity=0.5, marker=:square, markersize=2)
savefig("plot.png")

predict_label(X) = Flux.onecold(m.classifier(m.bagmodel(X)), classes)

cm, df = confusion_matrix(classes, Xk, yk, predict_label)
cm, df = confusion_matrix(classes, Xu, yu, predict_label)
cm, df = confusion_matrix(classes, Xt, yt, predict_label)

### reconstruction
function reconstruct(m::ChamferKLModel, Xb, y::Int, c)
    X = project_data(Xb)
    n = size(X, 2)
    x = m.bagmodel(Xb)
    yoh = Flux.onehot(y, 1:c)
    qc = condition(m.encoder, vcat(x, yoh))
    context = rand(qc)

    cy = vcat(context, yoh)
    Z = mapreduce(i -> rand(condition(m.generator, cy)), hcat, 1:n)

    Xhat = m.decoder(Z)
end
reconstruct(Xb, y) = reconstruct(m, Xb, y, c)

plt = []
for k in 1:12
    # i = sample(1:nobs(Xk))
    i = k
    Xb, y = Xk[i], ye[i]
    y = 2
    Xhat = reconstruct(Xb, y)
    p = scatter2(project_data(Xb), color=:3, xlims=(-3, 3), ylims=(-3, 3), axis=([], false), aspect_ratio=:equal, size=(400, 400), ms=project_data(Xb)[3, :] .+ 3 .* 1.5)
    p = scatter2!(Xhat, ms = Xhat[3, :] .+ 3 .* 1.5, opacity=0.7)
    push!(plt, p)
end
p = plot(plt..., layout=(3,4), size=(800,600))
safesave("plot_kld.png", p)

# on training data with known labels
enc = softmax(m.classifier(m.bagmodel(Xk)))
b = map(col -> any(col .> 0.9), eachcol(enc))
sum(b)
mean(Flux.onecold(enc[:, b], classes) .== yk[b])

# on test data
enc = softmax(m.classifier(m.bagmodel(Xt)))
b = map(col -> any(col .> 0.9), eachcol(enc))
sum(b)
mean(Flux.onecold(enc[:, b], classes) .== yt[b])

function reconstruction_accuracy(Xb)
    Xhat = map(yi -> reconstruct(Xb, yi), 1:c)
    ynew = findmin(x -> chamfer_distance(Xb.data.data, x), Xhat)[2]
end
mean(map(i -> reconstruction_accuracy(Xk[i]), 1:nobs(Xk)) .== ye)