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

using Plots
gr(markerstrokewidth=0, color=:jet, label="");

using Mill

using master_thesis: reindex, seqids2bags, encode

include(srcdir("point_cloud.jl"))
# project_data(X::AbstractBagNode) = Mill.data(Mill.data(X))

# load MNIST data and split it
data = load_mnist_point_cloud()
# b = map(x -> any(x .== [0,1,3,4]), data.bag_labels)
# filt_data, filt_labels = reindex(data.data, b), data.bag_labels[b]
r = 0.002
ratios = (r, 0.5-r, 0.5)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.bag_labels; ratios=ratios)
# Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(filt_data, filt_labels; ratios=ratios)
length(yk)
countmap(yk)
classes = sort(unique(yk))
n = c = length(classes)

# get loss functions
# using master_thesis.Models
# include(scriptsdir("M2bagmodel_loss.jl"))

# test models
Xb, yb = Xk[10], encode(yk, classes)[10]

model = M2_bag_constructor(Xk, c; bdim = 8, hdim=32, zdim=8, activation=relu)
semisupervised_loss(model, Xb, yb, Xb, c, 100)
semisupervised_loss_Chamfer(model, Xb, yb, Xb, c, 100)

model = M2_bag_constructor(Xk, c; bdim = 8, hdim=32, zdim=8, activation=relu)
semisupervised_loss(model, Xb, yb, Xb, c, 100)
semisupervised_loss_Chamfer(model, Xb, yb, Xb, c, 100)

model = M2_bag_constructor(Xk, c; hdim=16, zdim=4, type=:simple)
semisupervised_loss(model, Xb, yb, Xb, c, 100)
semisupervised_loss_Chamfer(model, Xb, yb, Xb, c, 100)

batchsize = 128
function minibatch()
    kix = sample(1:nobs(Xk), batchsize)
    uix = sample(1:nobs(Xu), batchsize)

    # encode labels to 1:c
    ye = encode(yk, classes)

    xk, y = [Xk[i] for i in kix], ye[kix]
    xu = [Xu[i] for i in uix]
    return xk, y, xu
end
function minibatch2()
    kix = sample(1:nobs(Xk), batchsize)
    uix = sample(1:nobs(Xu), batchsize)

    # encode labels to 1:c
    ye = encode(yk, classes)

    xk, y = reindex(Xk, kix), ye[kix]
    xu = reindex(Xu, uix)
    return xk, y, xu
end

loss(xk, yk, xu) = semisupervised_loss(model, xk, yk, xu, c, 0.1f0 * size(project_data(Xk), 2))
loss(xk, yk, xu) = semisupervised_loss_Chamfer(model, xk, yk, xu, c, 0.1f0 * size(project_data(Xk), 2))

function accuracy(model::M2BagModel, X, y, classes)
    N = length(y)
    ye = encode(y, classes)
    ynew = Flux.onecold(probs(condition(model.qy_x, model.bagmodel(X))), 1:length(classes))
    sum(ynew .== ye)/N
end
accuracy(X, y) = accuracy(model, X, y, classes)

tr_batch = minibatch();
mean(loss.(tr_batch...))

tr_batch = minibatch2();
lossf(tr_batch...)

opt = ADAM(0.001)
ps = Flux.params(model)
max_accuracy = 0
best_model = deepcopy(model)

k = 1
anim = @animate for i in k:k+99

    # scatter2(model.bagmodel(Xt), zcolor=yt, ms=2, label="no. $i")

    # b = map(_ -> minibatch_uniform(), 1:1)
    # b = minibatch()
    # Flux.train!(loss, ps, zip(b...), opt)
    b = minibatch2()
    Flux.train!(lossf, ps, [b], opt)

    @info "Epoch $i"
    # @show loss(tr_batch...)
    # @show mean(loss.(tr_batch...))
    @show accuracy(Xt, yt)
    @show a = accuracy(Xk, yk)
    a >= max_accuracy ? (global max_accuracy, best_model = a, deepcopy(model)) : nothing
end
k += 100
gif(anim, "animation4.gif", fps=10)



@time scatter2(best_model.bagmodel(Xk), zcolor=yk, ms=4, label="1")
@time scatter2(best_model.bagmodel(Xt), zcolor=yt, ms=2, label="1")

i = 0
i += 1; scatter2(reconstruct(best_model, Xk[i], encode(yk, classes)[i], c), label=yk[i], xlims=(-3.5,3.5), ylims=(-3.5,3.5))
i += 1; scatter2(reconstruct_mean(best_model, Xk[i], encode(yk, classes)[i], c), label=yk[i], xlims=(-3.5,3.5), ylims=(-3.5,3.5))

i += 1; scatter2(reconstruct(best_model, Xt[i], encode(yt, classes)[i], c), label=yt[i], xlims=(-3.5,3.5), ylims=(-3.5,3.5))

# accuracy of predictions

predict_label(X) = Flux.onecold(probs(condition(best_model.qy_x, best_model.bagmodel(X))), classes)

cm, df = confusion_matrix(classes, Xk, yk, predict_label)
cm, df = confusion_matrix(classes, Xt, yt, predict_label)

using UMAP
emb = umap(best_model.bagmodel(Xk), 2)
scatter2(emb, zcolor=yk, ms=4)

emb = umap(best_model.bagmodel(Xt)[:, 1:4000], 2)
scatter2(emb, zcolor=yt[1:4000], ms=3)

### mean over minibatches
# assume Xb is a BagNode with batchsize nodes
# y is a vector of bag labels

# this splits the BagNode into individual bags without tracking it for differentiation
Xb = Xk[1:10]
yb = encode(yk, classes)[1:10]
nbags = nobs(Xb)
bagvec = Flux.Zygote.@ignore [Xb[i] for i in 1:nbags]

# now we should be able to dispatch over bags and labels
function lossf(Xk, yk, Xu)
    nk = nobs(Xk)
    bk = Flux.Zygote.@ignore [Xk[i] for i in 1:nk]

    nu = nobs(Xu)
    bu = Flux.Zygote.@ignore [Xu[i] for i in 1:nu]
    mean(loss.(bk, yk, bu))
end


### view more digits
plot_number(i) = scatter2(project_data(Xk[i]), aspect_ratio=:equal, ms=(project_data(Xk[i])[3, :] .+ 3) .* 2, size=(400, 400), marker=:square)

function plot9()
    k = sample(1:length(yk)-9)
    pvec = []
    for i in k:k+8
        p_i = plot_number(i)
        yi = encode([yk[i]], classes)[1]
        R = reconstruct(best_model, Xk[i], encode(yk, classes)[i], c)
        r = classes[Flux.onecold(condition(best_model.qy_x, best_model.bagmodel(Xk[i])).α)][1]
        p_i = scatter2!(
            R, ms=R[3, :] .+ 3 .* 2,
            color=:green, axis=([], false), xlims=(-3,3), ylims=(-3, 3), size=(900, 900),
            title="predicted: $r\ntrue: $(yk[i])", titlefontsize=8
        )
        push!(pvec, p_i)
    end
    plot(pvec..., layout=(3,3), size=(900,900))
end

function plot9_test()
    plot_number(i) = scatter2(
        project_data(Xt[i]), aspect_ratio=:equal,
        ms=(project_data(Xt[i])[3, :] .+ 3) .* 2,
        size=(400, 400), marker=:square,
        opacity=0.6, color=:grey
    )
    
    k = sample(1:length(yt)-9)
    pvec = []
    for i in k:k+8
        p_i = plot_number(i)
        r = classes[Flux.onecold(condition(best_model.qy_x, best_model.bagmodel(Xt[i])).α)][1]
        # r = 5
        R = reconstruct(best_model, Xt[i], encode([r], classes)[1], c)
        # R = reconstruct(best_model, Xt[i], encode(yt, classes)[i], c)
        p_i = scatter2!(
            R, ms=R[3, :] .+ 3 .* 2,
            color=:green, axis=([], false), xlims=(-3,3), ylims=(-3, 3), size=(900, 900),
            title="predicted: $r\ntrue: $(yt[i])", titlefontsize=8
        )
        push!(pvec, p_i)
    end
    plot(pvec..., layout=(3,3), size=(900,900))
end

plot9_test()