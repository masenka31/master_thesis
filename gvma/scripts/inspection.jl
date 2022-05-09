using DrWatson
@quickactivate
using gvma
using gvma: encode
include(srcdir("init_strain.jl"))

using Flux3D: chamfer_distance

r = 0.01
ratios = (r, 0.5-r, 0.5)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(X, y; ratios=ratios, seed=1)
classes = sort(unique(yk))
c = length(classes)

model = reflectinmodel(
    Xk,
    k -> Dense(k, 2),
    BagCount ∘ SegmentedMeanMax
);

model(Xk);

keys(model)
# this is a submodel = BagModel
model[:behavior_summary][:timers]
model[:signatures]
model[:network_http]
# 

x = Xk[1]
x1 = x[:behavior_summary]
pn_keys = keys(x1)

xt = x1[:read_files]
using Mill.SparseArrays
m = sparse(xt.data.data)
mdense = Float32.(Matrix(m))

xt2 = Xk[2][:behavior_summary][:read_files]
y2 = yk[2]
xt3 = Xk[13][:behavior_summary][:read_files]
m2 = sparse(xt2.data.data)
m3 = sparse(xt3.data.data)

md2 = Matrix(m2)
md3 = Matrix(m3)

bm = BagModel(Dense(2053, 64, swish), BagCount(SegmentedMeanMax(64)), Chain(Dense(64*2+1, 32, swish), Dense(32, 10)))
bm(xt2)
xdim, n = size(xt2.data)
context = vcat(bm(xt2), Flux.onehot(y2, classes))
h = repeat(context, outer=(1, n))

using ConditionalDists, DistributionsAD, Distributions

idim = length(context)
hdim = 64
zdim = 32
gen = Chain(
            Dense(idim, hdim, swish), Dense(hdim, hdim, swish),
            SplitLayer(hdim,[zdim,zdim])
)
gen_dist = ConditionalMvNormal(gen)

z = rand(condition(gen_dist, h))

decoder = Chain(Dense(zdim, hdim, swish), Dense(hdim, hdim, swish), Dense(hdim, xdim, relu))
xhat = decoder(z)

Flux.mse(m2 * 100, xhat)

function loss(x, y, classes)
    n = Flux.Zygote.@ignore size(x.data)[2]
    m = Flux.Zygote.@ignore Matrix(sparse(x.data.data))

    context = vcat(bm(x), Flux.onehot(y, classes))
    h = repeat(context, outer=(1, n))
    z = rand(condition(gen_dist, h))
    xhat = decoder(z)
    
    Flux.mse(m * 100, xhat * 100)
end
function loss_enc(x, y, classes)
    if isempty(x)
        return 0f0
    end
    n = Flux.Zygote.@ignore size(x.data)[2]
    m = Flux.Zygote.@ignore Matrix{Float32}(sparse(x.data.data))
    yoh = Float32.(repeat(Flux.onehot(y, classes), outer=(1,n)))
    enc = instance_encoder(vcat(m, yoh))

    context = vcat(bm(x), Flux.onehot(y, classes))
    h = repeat(context, outer=(1, n))
    henc = vcat(h, enc)
    z = rand(condition(gen_dist, henc))
    xhat = decoder(z)
    
    Flux.mse(m * 100, xhat * 100)
end
lossf(x, y) = loss(x, y, classes)
lossf(x, y) = loss_enc(x, y, classes)

lossf(x, y) = isempty(x) ? (return loss_enc(x, y, classes)) : (return 0f0)

batch_loss(x, y) = mean(lossf.(x, y))

# check that differentiation works
l() = loss_enc(xt2, y2, classes)
l() = batch_loss(xdata, yk)
ps = Flux.params(bm, gen_dist, decoder, instance_encoder)
g = Flux.gradient(l, ps)

# the training data cannot contain empty bags!
xdata = map(i -> Xk[i][:behavior_summary][:read_files], 1:nobs(Xk))
len = map(x -> size(x.data)[2], xdata)
b = len .!= 0
xd = xdata[b]
yd = yk[b]

bm = BagModel(Dense(2053, 64, swish), BagCount(SegmentedMeanMax(64)), Chain(Dense(64*2+1, 32, swish), Dense(32, 10)))
gen = Chain(
            Dense(idim+32, hdim, swish), Dense(hdim, hdim, swish),
            SplitLayer(hdim,[zdim,zdim])
)
gen_dist = ConditionalMvNormal(gen)
decoder = Chain(Dense(zdim, hdim, swish), Dense(hdim, hdim, swish), Dense(hdim, xdim, relu))
instance_encoder = Chain(Dense(2053+c, hdim, swish), Dense(hdim, hdim, swish), Dense(hdim, 32))

lossf(xd[1], yd[2])
opt = ADAM()
ps = Flux.params(bm, gen_dist, decoder, instance_encoder)
Flux.@epochs 100 begin
    for i in 1:sum(b)
        Flux.train!(lossf, ps, repeated((xd[i], yd[i]), 1), opt)
    end
    @show mean(lossf.(xd, yd))
end

opt = ADAM()
ps = Flux.params(bm, gen_dist, decoder, instance_encoder)
Flux.@epochs 100 begin
    Flux.train!(batch_loss, ps, repeated((xd, yd), 10), opt)
    @show mean(lossf.(xd, yd))
end


function score(x, y, classes; type=rand)
    n = Flux.Zygote.@ignore size(x.data)[2]
    m = Flux.Zygote.@ignore Matrix{Float32}(sparse(x.data.data))
    yoh = Float32.(repeat(Flux.onehot(y, classes), outer=(1,n)))
    enc = instance_encoder(vcat(m, yoh))

    context = vcat(bm(x), Flux.onehot(y, classes))
    h = repeat(context, outer=(1, n))
    henc = vcat(h, enc)
    z = type(condition(gen_dist, henc))
    xhat = round.(decoder(z))
    
    Flux.mse(m, xhat)
end
score_rand(x, y) = score(x, y, classes)
score_mean(x, y) = score(x, y, classes; type=mean)

mean(_ -> mean(map(x -> classes[findmin(c -> lossf(x, c), classes)[2]], xd) .== yd), 1:20)
# ynew = map(x -> classes[findmin(c -> lossf(x, c), classes)[2]], xd)
# mean(ynew .== yd)

ynew = map(x -> classes[findmin(c -> score_rand(x, c), classes)[2]], xd)
mean(ynew .== yd)

ynew = map(x -> classes[findmin(c -> score_mean(x, c), classes)[2]], xd)
mean(ynew .== yd)

# test data?
xtt = map(i -> Xt[i][:behavior_summary][:read_files], 1:nobs(Xt))
len = map(x -> size(x.data)[2], xtt)
b = len .!= 0
xtest = xtt[b]
ytest = yt[b]

mean(_ -> mean(map(x -> classes[findmin(c -> lossf(x, c), classes)[2]], xtest) .== ytest), 1:10)

# create data as a dictionary
