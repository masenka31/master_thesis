using DrWatson
@quickactivate
using gvma
using gvma: encode
include(srcdir("init_strain.jl"))

using Base.Iterators: repeated

r = 0.01
ratios = (r, 0.5-r, 0.5)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(X, y; ratios=ratios, seed=1)
classes = sort(unique(yk))
c = length(classes)

Xd = X[:behavior_summary]
ks = keys(Xd)

parameters = (hdim=256, cdim=32, zdim=32)
models = map(key -> model_constructor(Xd, key, c; parameters...), ks)
ensemble = Dict(ks .=> models)

# train one submodel
xdata = [Xk[:behavior_summary][i] for i in 1:nobs(Xk)]

# filter keys -> it does not make sense to train on empty data
f = map(k -> !isempty(Xk[:behavior_summary][k]), ks) |> BitArray
fks = ks[f]
fmodels = filter(model -> in(model.key, fks), models)

batchsize=64
function minibatch()
    ix = sample(1:nobs(Xk), batchsize)
    x = xdata[ix]
    y = yk[ix]
    return x, y
end

loss(x, y) = sum(map(model -> loss_enc(model, x, y, classes), fmodels))
batch_loss(x, y) = mean(loss.(x, y))
# loss(x, y) = loss_enc(model, x, y, classes)
opt = ADAM()
ps = Flux.params(fmodels);
Flux.@epochs 100 begin
    batches = map(_ -> minibatch(), 1)
    # Flux.train!(batch_loss, ps, repeated((xdata, yk), 10), opt)
    Flux.train!(batch_loss, ps, (batches,), opt)
    @show batch_loss(xdata, yk)
end

max_train_time = 60*60
start_time = time()
while time() - start_time < max_train_time
    batches = map(_ -> minibatch(), 1)
    # Flux.train!(batch_loss, ps, repeated((xdata, yk), 10), opt)
    Flux.train!(batch_loss, ps, (batches,), opt)
    @show batch_loss(xdata, yk)
end

m = mapreduce(_ -> map(c -> loss(xdata[1], c), classes), hcat, 1:10)
findmin.(eachcol(m))

using gvma: my_findmax
function cfindmin(col)
    min = minimum(col)
    ix = findall(x -> x == min, col)
    length(ix) == 1 ? findmin(col) : (0, length(col)+1)
end
cfindmin(fun, itr) = cfindmin(map(i -> fun(i), itr))
function _majority_vote(ensemble, ks, classes, Xt)
    uclasses = vcat(classes, "undecided")
    xtdata = [Xt[:behavior_summary][i] for i in 1:nobs(Xt)]
    labels = Matrix{String}(undef, length(ks), nobs(Xt))
    for (i, key) in enumerate(ks)
        sc = mapreduce(x -> uclasses[cfindmin(c -> loss_enc(ensemble[key], x, c, classes), classes)[2]], hcat, xtdata)
        labels[i, :] = sc
    end
    return labels
end
function majority_vote(ensemble, ks, classes, Xt, yt)
    L = majority_vote(ensemble, ks, classes, Xt)
    r = map(x -> countmap(x), eachcol(L))
    map(x -> pop!(x, "undecided"), r)
    ynew = map(x -> findmax(x)[2], r)
    a = mean(ynew .== yt)

    ytry = map(x -> my_findmax(x), r)
    b = length.(ytry) .== 1
    ab = mean(ynew[b] .== yt[b])

    @info """Accuracy is:
    $(round(a, digits=3)) for full data
    $(length(r)-sum(b)) data undecided
    $(round(ab, digits=3)) for decided data
    """

    return ynew, (ytry, b)
end

