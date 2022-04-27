using DrWatson
@quickactivate
using gvma
using gvma: encode
include(srcdir("init_strain.jl"))

r = 0.01
ratios = (r, 0.5-r, 0.5)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(X, y; ratios=ratios, seed=1)
classes = sort(unique(yk))
c = length(classes)

Xd = X[:behavior_summary]
ks = keys(Xd)

models = map(key -> model_constructor(Xd, key, c), ks)
ensemble = Dict(ks .=> models)
copy_opt = ADAM()
opts = Dict(ks .=> repeat([deepcopy(copy_opt)], length(ks)))

# train one submodel
xdata = [Xk[:behavior_summary][i] for i in 1:nobs(Xk)]

f = map(k -> !isempty(Xk[:behavior_summary][k]), ks) |> BitArray
# filtered keys -> it does not make sense to train on empty data
fks = ks[f]

accuracies = Dict(ks .=> repeat([[]], length(ks)))

function train_single(model, xdata, yk, o, classes, accuracies; iters=10)
    lossf(x, y) = loss_enc(model, x, y, classes)
    batch_loss(x, y) = mean(lossf.(x, y))

    ps = Flux.params(model);
    opt = deepcopy(o)

    @info "Starting $iters epochs on $(model.key) subset."
    Flux.@epochs iters begin
        Flux.train!(batch_loss, ps, repeated((xdata, yk), 10), opt)
        mean_loss = mean(filter(x -> x != 0, lossf.(xdata, yk)))
        @show mean_loss
        # mean_acc = round(mean(_ -> mean(map(x -> classes[findmin(c -> lossf(x, c), classes)[2]], xdata) .== yk), 1:2), digits=3)
        # @show mean_acc
    end
    mean_acc = round(mean(_ -> mean(map(x -> classes[findmin(c -> loss_enc(model, x, c, classes), classes)[2]], xdata) .== yk), 1:10), digits=3)
    @info "Mean accuracy of $mean_acc for $(model.key)."

    return model, opt, deepcopy(mean_acc)
end

Flux.@epochs 20 begin
    for key in fks
        model = ensemble[key]
        opt = opts[key]
        m, o, a = train_single(model, xdata, yk, opt, classes, accuracies; iters=5)
        avec = accuracies[key]
        avec = vcat(avec, a)
        accuracies[key] = avec
        opts[key] = o
    end
end


scores = minimum_score_over_classes(ensemble, fks, classes, Xt)
ynew = Flux.onecold(.- scores, classes)
mean(ynew .== yt)

scores = minimum_score_over_classes_normalized(ensemble, fks, classes, Xt, Xk)
ynew = Flux.onecold(.- scores, classes)
mean(ynew .== yt)

test_loss(x, y) = loss_enc(ensemble[:files], x, y, classes)
mean_acc = round(mean(_ -> mean(map(x -> classes[findmin(c -> test_loss(x, c), classes)[2]], xdata) .== yk), 1:20), digits=3)
xtdata = [Xt[:behavior_summary][i] for i in 1:nobs(Xt)]
mean_acc = round(mean(_ -> mean(map(x -> classes[findmin(c -> test_loss(x, c), classes)[2]], xtdata) .== yt), 1:5), digits=3)

function minimum_score_over_classes(ensemble, ks, classes, Xt)
    xtdata = [Xt[:behavior_summary][i] for i in 1:nobs(Xt)]
    scores = zeros(length(classes), nobs(Xt))
    for key in ks
        sc = mapreduce(x -> map(c -> loss_enc(ensemble[key], x, c, classes), classes), hcat, xtdata)
        scores = scores .+ sc
    end
    return scores
end
function minimum_score_over_classes_normalized(ensemble, ks, classes, Xt, Xk)
    xdata = [Xt[:behavior_summary][i] for i in 1:nobs(Xk)]
    xtdata = [Xt[:behavior_summary][i] for i in 1:nobs(Xt)]
    scores = zeros(length(classes), nobs(Xt))
    for key in ks
        sck = mapreduce(x -> map(c -> loss_enc(ensemble[key], x, c, classes), classes), hcat, xdata)
        if unique(sck) == [0.0]
            # skip
            @info "Key $key would result in NaNs, skipping..."
        else
            f = fit(UnitRangeTransform, sck)
            sct = mapreduce(x -> map(c -> loss_enc(ensemble[key], x, c, classes), classes), hcat, xtdata)
            sc = StatsBase.transform(f, sct)
            scores = scores .+ sc
        end
    end
    return scores
end
function majority_vote(ensemble, ks, classes, Xt)
    uclasses = vcat(classes, "undecided")
    xtdata = [Xt[:behavior_summary][i] for i in 1:nobs(Xt)]
    labels = Matrix{String}(undef, length(ks), nobs(Xt))
    for (i, key) in enumerate(ks)
        sc = mapreduce(x -> uclasses[cfindmin(c -> loss_enc(ensemble[key], x, c, classes), classes)[2]], hcat, xtdata)
        labels[i, :] = sc
    end
    return labels
end
r = map(x -> countmap(x), eachcol(L))
map(x -> pop!(x, "undecided"), r)
ynew = map(x -> findmax(x)[2], r)
ytry = map(x -> my_findmax(x), r)
b = length.(ytry) .== 1
mean(ynew[b] .== yt[b])


test_loss(x, y) = loss_enc(ensemble[:files], x, y, classes)
test_loss(xdata[1], "Kraton")
# batch_loss(x, y) = mean(lossf.(x, y))
mean_acc = round(mean(_ -> mean(map(x -> classes[findmin(c -> test_loss(x, c), classes)[2]], xdata) .== yk), 1:20), digits=3)

function cfindmin(col)
    min = minimum(col)
    ix = findall(x -> x == min, col)
    length(ix) == 1 ? findmin(col) : (0, length(col)+1)
end
cfindmin(fun, itr) = cfindmin(map(i -> fun(i), itr))

function cfindmax(col)
    min = maximum(col)
    ix = findall(x -> x == min, col)
    length(ix) == 1 ? findmax(col) : (0, length(col)+1)
end
cfindmax(fun, itr) = cfindmax(map(i -> fun(i), itr))

test_data = [Xt[:behavior_summary][i] for i in 1:nobs(Xt)]
mean_acc = round(mean(_ -> mean(map(x -> classes[cfindmin(c -> test_loss(x, c), classes)[2]], test_data) .== yt), 1:2), digits=3)


struct Ensemble
    keys::NTuple
    models::Dict
    opts::Dict
end

models = map(key -> model_constructor(Xd, key, c), ks)
ensemble = Dict(ks .=> models)
opts = Dict(ks .=> repeat([ADAM()], length(ks)))
xdata = [Xk[:behavior_summary][i] for i in 1:nobs(Xk)]


for key in ks
    model = ensemble[key]
    opt = opts[key]
    lossf1(x, y) = loss_enc(ensemble[key], x, y, classes)
    batch_loss1(x, y) = mean(lossf1.(x, y))
    Flux.train!(batch_loss1, Flux.params(ensemble[key]), repeated((xdata, yk), 1), opts[key])
    mean_acc = round(mean(_ -> mean(map(x -> classes[findmin(c -> lossf1(x, c), classes)[2]], xdata) .== yk), 1:2), digits=3)
    ensemble[key] = model
    opts[key] = opt
    @show (key, mean_acc)
end

psprev = deepcopy(Flux.params(ensemble[:timers]))