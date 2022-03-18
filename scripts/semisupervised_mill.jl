using Mill
include(srcdir("toy", "moons.jl"))
gr(markerstrokewidth=0, color=:jet);

# get moons array data
n = 10
data, y, lnum = generate_moons_data(n, 300; λ=50, gap=1, max_val = 10)
X = hcat(data...)
scatter2(X, color=lnum,aspect_ratio=:equal, ms=2.5)

# create mill data
mill_data = BagNode(ArrayNode(hcat(data...)), get_obs(data))

##################################################
###                 Classifier                 ###
##################################################

# split known data to train/test
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_data(mill_data, y, ratios=(0.1, 0.4, 0.5))

# and encode labels to onehot
Xtrain = Xk
ytrain = yk
yoh_train = Flux.onehotbatch(ytrain, 1:n)

# create a simple classificator model
mill_model = reflectinmodel(
    Xtrain,
    d -> Dense(d, 2)
)
model = Chain(mill_model, Mill.data, Dense(2, n))

# training parameters, loss etc.
opt = ADAM()
loss(x, y) = Flux.logitcrossentropy(model(x), y)
accuracy(x, y) = round(mean(collect(1:n)[Flux.onecold(model(x))] .== y), digits=3)

using IterTools
using Flux: @epochs

@epochs 100 begin
    Flux.train!(loss, Flux.params(model), repeated((Xtrain, yoh_train), 5), opt)
    @show loss(Xtrain, yoh_train)
    @show accuracy(Xtrain, ytrain)
    @show accuracy(Xt, yt)
end

# look at the created latent space
XX, yy = Xtrain, ytrain
latent = model[1:end-1](XX)
Xtdata = XX.data.data
ytdata = mapreduce((y, i) -> repeat([y], i), vcat, yy, length.(XX.bags))
scatter2(Xtdata, zcolor=ytdata)
scatter2(latent, zcolor=yy)

#######################################################
###                 Semi-supervised                 ###
#######################################################

# mill model to get one-vector bag representation
bagmodel = Chain(reflectinmodel(
    Xk,
    d -> Dense(d, 2),
    SegmentedMeanMax
), Mill.data, softmax)

project_data(X::AbstractBagNode) = Mill.data(Mill.data(X))

# parameters
c = n       # number of classes
dz = 2      # latent dimension
xdim = 2    # input dimension

# latent prior - isotropic gaussian
pz = MvNormal(zeros(dz), 1)

# categorical prior
α = softmax(randn(c))   # α is a trainable parameter!
py = Categorical(α)

# posterior on latent (known labels)
μ_qz_xy = Chain(Dense(xdim+c,2,swish), Dense(2,dz))
σ_qz_xy = Chain(Dense(xdim,2,swish), Dense(2,dz,softplus))
qz_xy(x) = DistributionsAD.TuringDiagMvNormal(μ_qz_xy(x), σ_qz_xy(x[1:xdim]))

# posterior on y (needs the bag aggregation)
α_qy_x = Chain(Dense(2,2,swish), Dense(2,c),softmax)
qy_x(B) = Categorical(α_qy_x(bagmodel(B)[:]))

# posterior on x
μ_px_yz = Chain(Dense(xdim+c,2,swish), Dense(2,xdim))
σ_px_yz = Chain(Dense(xdim+c,2,swish), Dense(2,xdim,softplus))
px_yz(x) = DistributionsAD.TuringDiagMvNormal(μ_px_yz(x), σ_px_yz(x))

# parameters and opt
ps = Flux.params(α, μ_qz_xy, σ_qz_xy, α_qy_x, μ_px_yz, σ_px_yz, bagmodel)
opt = ADAM()

# minibatch function for bags
function minibatch(Xk, y, Xu;ksize=64, usize=64)
    kix = sample(1:nobs(Xk), ksize)
    uix = sample(1:nobs(Xu), usize)

    xk, yk = [Xk[i] for i in kix], y[kix]
    xu = [Xu[i] for i in uix]

    return xk, yk, xu
end

function accuracy(X, y, c)
    N = length(y)
    ynew = map(i -> Flux.onecold(qy_x(X[i]).p, 1:c), 1:nobs(X))
    sum(ynew .== y)/N
end
accuracy(X, y) = accuracy(X, y, c)

# semisupervised loss stays the same
ksize, usize = 64, 64
loss(xk, yk, xu) = semisupervised_loss(xk, yk, xu, ksize)

for i in 1:20
    b = minibatch(Xk, yk, Xu; ksize=ksize, usize=usize);
    Flux.train!(loss, ps, zip(b...), opt)
    @show i
    a = round(accuracy(Xk, yk), digits=4)
    @show a
    @show loss(b[1][1], b[2][1], b[3][1])
end

r = mapreduce(i -> qy_x(Xt[i]).p, hcat, 1:nobs(Xt))
bar(r[1,:], color=Int.(yt), size=(1000,400))