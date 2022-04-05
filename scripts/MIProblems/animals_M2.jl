using DrWatson
@quickactivate

using master_thesis
using Flux

include(srcdir("point_cloud.jl"))
include(srcdir("mill_data.jl"))

function load_animals()
    fox = load_mill_data("Fox").normal
    elephant = load_mill_data("Elephant").normal
    tiger = load_mill_data("Tiger").normal

    yf = repeat(["Fox"], nobs(fox))
    ye = repeat(["Elephant"], nobs(elephant))
    yt = repeat(["Tiger"], nobs(tiger))

    return cat(fox, elephant, tiger), vcat(yf, ye, yt)
end

data, labels = load_animals()

r = 0.02
ratios = (r, 0.5-r, 0.5)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data, labels; ratios=ratios, seed=1)

classes = sort(unique(yk))
n = c = length(classes)

using master_thesis.Models
using master_thesis: encode

function accuracy(model::M2BagModel, X, y, classes)
    ynew = Flux.onecold(model.bagmodel(X), classes)
    mean(ynew .== y)
end
accuracy(X, y) = accuracy(model, X, y, classes)
@show accuracy(Xk, yk)


ye = encode(yk, classes)

batchsize = 64
function minibatch()
    kix = sample(1:nobs(Xk), batchsize)
    uix = sample(1:nobs(Xu), batchsize)

    xk, y = reindex(Xk, kix), ye[kix]
    xu = reindex(Xu, uix)
    return xk, y, xu
end

# lknown(xk, y) = master_thesis.Models.loss_known_bag_Chamfer(model, xk, y, c)
# lunknown(xu) = master_thesis.Models.loss_unknown_Chamfer(model, xu, c)
lknown(xk, y) = master_thesis.Models.loss_known_bag(model, xk, y, c)
lunknown(xu) = master_thesis.Models.loss_unknown(model, xu, c)

function loss_rec(Xk, yk, Xu)
    l_known = mean(lknown.(Xk, yk))
    l_unknown = mean(lunknown.(Xu))
    return l_known + l_unknown
end

N = size(project_data(Xk), 2)
α = 0.1f0
lclass(x, y) = loss_classification_crossentropy(model, x, y, c) * α * N

function lossf(Xk, yk, Xu)
    nk = nobs(Xk)
    bk = Flux.Zygote.@ignore [Xk[i] for i in 1:nk]

    nu = nobs(Xu)
    bu = Flux.Zygote.@ignore [Xu[i] for i in 1:nu]
    
    lr = loss_rec(bk, yk, bu)
    lc = lclass(Xk, yk)
    return lr + lc
end
function lossf_warmup(Xk, yk, Xu)
    nk = nobs(Xk)
    bk = Flux.Zygote.@ignore [Xk[i] for i in 1:nk]

    lr = mean(lknown.(bk, yk))
    lc = lclass(Xk, yk)
    return lc + lr
end


model = M2_bag_constructor(Xk, c; bdim=8, hdim=16, zdim=8, type=:dense)

@show lossf(minibatch()...)
@show lossf_warmup(minibatch()...)

opt = ADAM()
ps = Flux.params(model)
max_accuracy = 0
best_model = deepcopy(model)
max_train_time = 180

# for i in 1:100
#     b = map(i -> minibatch(), 1:5)
#     Flux.train!(lossf_warmup, ps, b, opt)
#     @show a = accuracy(Xk, yk)
# end

start_time = time()
while time() - start_time < max_train_time

    b = map(i -> minibatch(), 1:5)
    Flux.train!(lossf, ps, b, opt)
    @show lossf(minibatch()...)

    if isnan(lossf(minibatch()...))
        @info "Loss in NaN, stopped training, moving on..."
        break
    end

    @show a = accuracy(Xk, yk)
    if a >= max_accuracy
        @show accuracy(Xt, yt)
        max_accuracy = a
        best_model = deepcopy(model)
    end
end

accuracy(Xk, yk)
accuracy(Xu, yu)
accuracy(Xt, yt)

using Plots
ENV["GKSwstype"] = "100"
gr(markerstrokewidth=0, color=:jet, label="");

scatter2(model.bagmodel(Xk), zcolor=encode(yk, classes), ms=4)
scatter2!(model.bagmodel(Xu), zcolor=encode(yu, classes), opacity=0.5, marker=:square, markersize=2)
scatter2!(model.bagmodel(Xt), zcolor=encode(yt, classes), marker=:star)

predict_label(X) = Flux.onecold(model.bagmodel(X), classes)
predict_label(X) = Flux.onecold(best_model.bagmodel(X), classes)

cm, df = confusion_matrix(classes, Xu, yu, predict_label)
cm, df = confusion_matrix(classes, Xt, yt, predict_label)