using DrWatson
@quickactivate
using gvma
using gvma: encode
include(srcdir("init_strain.jl"))

r = 0.01
ratios = (r, 0.5-r, 0.5)
seed = 1
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(X, y; ratios=ratios, seed=seed)
classes = sort(unique(yk))
c = length(classes)
Xval, yval, Xu, yu = validation_data(yk, Xu, yu, seed, classes)

Xd = X[:behavior_summary]
ks = keys(Xd)

# prepare data for training
xdata = [Xk[:behavior_summary][i] for i in 1:nobs(Xk)]
xvdata = [Xval[:behavior_summary][i] for i in 1:nobs(Xval)]
xudata = [Xu[:behavior_summary][i] for i in 1:nobs(Xu)]
xtdata = [Xt[:behavior_summary][i] for i in 1:nobs(Xt)]

# filter keys -> it does not make sense to train on empty data
f = map(k -> !isempty(Xk[:behavior_summary][k]), ks) |> BitArray
fks = ks[f]
parameters = (hdim=128, cdim=32, zdim=32, bdim=32)
model = M2constructor(Xd, ks, c; parameters...)
# models = map(key -> probmodel_constructor(Xd, key, c; parameters...), fks)
# fmodels = filter(model -> in(model.key, fks), models)
# ensemble = Dict(ks .=> models)

batchsize=64
ye = encode(yk, classes)
function minibatch()
    ixk = sample(1:nobs(Xk), batchsize)
    ixu = sample(1:nobs(Xu), batchsize)
    x = xdata[ixk]
    y = ye[ixk]
    xu = xudata[ixu]
    return x, y, xu
end
test_batch = minibatch()

α = 0.1f0 * nobs(Xu)
loss(xk, yk, xu) = semisupervised_loss(model, xk, yk, xu, c, α)
batch_loss(x, y, xu) = mean(loss.(x, y, xu))

@show batch_loss(test_batch...)
opt = ADAM()
ps = Flux.params(model);

Flux.@epochs 10 begin
    batch = minibatch()
    # Flux.train!(batch_loss, ps, repeated((xdata, yk), 10), opt)
    Flux.train!(batch_loss, ps, (batch,), opt)
    @show batch_loss(test_batch...)
end

test_batch = minibatch()
best_val = 0
best_tr = 0
best_model = deepcopy(model)
max_train_time = 60*10
start_time = time()
while time() - start_time < max_train_time
    batch = minibatch()
    # Flux.train!(batch_loss, ps, repeated((xdata, yk), 10), opt)
    Flux.train!(batch_loss, ps, (batch,), opt)
    @show batch_loss(test_batch...)
    @show tr = mean(predictions(Xk[:behavior_summary]) .== yk)
    @show val = mean(predictions(Xval[:behavior_summary]) .== yval)
    if (val >= best_val) && (tr >= best_tr)
        best_model = deepcopy(model)
        best_val = val
        best_tr = tr
        @info "Validation accuracy improved."
    end
end

using ConditionalDists
predictions(X) = Flux.onecold(condition(model.qy_x, model.bagmodel(X)).α, classes)
mean(predictions(Xk[:behavior_summary]) .== yk)
mean(predictions(Xu[:behavior_summary]) .== yu)
mean(predictions(Xt[:behavior_summary]) .== yt)