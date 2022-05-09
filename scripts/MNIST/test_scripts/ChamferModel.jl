using DrWatson
@quickactivate
using master_thesis
using master_thesis.Models
using master_thesis: reindex, seqids2bags, encode
using Flux, Mill

include(srcdir("point_cloud.jl"))

r = 0.002
ratios = (r, 0.5-r, 0.5)        # get the ratios
data = load_mnist_point_cloud()
seed = 1

b = map(x -> any(x .== [0,1,3,4]), data.bag_labels)
filt_data, filt_labels = reindex(data.data, b), data.bag_labels[b]
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(filt_data, filt_labels; ratios=ratios, seed=seed)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.bag_labels; ratios=ratios, seed=seed)

classes = sort(unique(yk))
n = c = length(classes)

function accuracy(m::ChamferModel, X, y, classes)
    ynew = Flux.onecold(m.classifier(m.bagmodel(X)), classes)
    mean(ynew .== y)
end
accuracy(X, y) = accuracy(m, X, y, classes)

lknown(xk, y) = loss_known(m, xk, y, c)
lunknown(xu) = loss_unknown(m, xu, c)

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

agg = BagCount ∘ SegmentedMeanMax
m = chamfermodel_constructor(Xk, c; cdim=8, hdim=32, bdim=8, gdim=8, aggregation=agg)

ps = Flux.params(m)
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

scatter2(m.bagmodel(Xk), zcolor=yk, ms=4)
scatter2!(m.bagmodel(Xu), zcolor=yu, opacity=0.5, marker=:square, markersize=2)
savefig("plot.png")

predict_label(X) = Flux.onecold(m.classifier(m.bagmodel(X)), classes)

cm, df = confusion_matrix(classes, Xk, yk, predict_label)
cm, df = confusion_matrix(classes, Xu, yu, predict_label)
cm, df = confusion_matrix(classes, Xt, yt, predict_label)

### reconstruction
# import master_thesis.Models: reconstruct
function reconstruct(m::ChamferModel, Xb, y::Int, c)
    X = project_data(Xb)
    n = size(X, 2)
    context = m.bagmodel(Xb)
    yoh = Flux.onehot(y, 1:c)

    Z = mapreduce(i -> vcat(rand(condition(m.generator, context)), yoh), hcat, 1:n)
    Xhat = m.decoder(Z)
end
reconstruct(Xb, y) = reconstruct(m, Xb, y, c)

plt = []
for k in 1:12
    # i = sample(1:nobs(Xk))
    i = k
    Xb, y = Xk[i], ye[i]
    # y = 1
    Xhat = reconstruct(Xb, y)
    p = scatter2(project_data(Xb), color=:3, xlims=(-3, 3), ylims=(-3, 3), axis=([], false), aspect_ratio=:equal, size=(400, 400), ms=project_data(Xb)[3, :] .+ 3 .* 1.5)
    p = scatter2!(Xhat, ms = Xhat[3, :] .+ 3 .* 1.5, opacity=0.7)
    push!(plt, p)
end
p = plot(plt..., layout=(3,4), size=(800,600))
wsave("plot_chamfer.png", p)

# on training data with known labels
enc = softmax(m.classifier(m.bagmodel(Xk)))
b = map(col -> any(col .> 0.9), eachcol(enc))
sum(b)
mean(Flux.onecold(enc[:, b], classes) .== yk[b])

# on test data
enc = softmax(m.classifier(m.bagmodel(Xt)))
b = map(col -> any(col .> 0.99), eachcol(enc))
sum(b)
mean(Flux.onecold(enc[:, b], classes) .== yt[b])

function reconstruction_accuracy(Xb)
    Xhat = map(yi -> reconstruct(Xb, yi), 1:c)
    ynew = findmin(x -> chamfer_distance(Xb.data.data, x), Xhat)[2]
end
mean(map(i -> reconstruction_accuracy(Xk[i]), 1:nobs(Xk)) .== ye)
mean(map(i -> reconstruction_accuracy(Xt[i]), 1:nobs(Xt)) .== encode(yt, classes))