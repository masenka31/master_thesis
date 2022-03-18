"""
	safe_softplus(x::T)

Safe version of softplus.	
"""
safe_softplus(x::T) where T  = softplus(x) + T(0.000001)

using ConditionalDists

# new conditional distributions
c = 3       # number of classes
zdim = 2    # latent dimension
xdim = 2    # input dimension

net_xz = Chain(Dense(xdim+c,4,swish), Dense(4, 4, swish), SplitLayer(4, [zdim,zdim], [identity, safe_softplus]))
net_zx = Chain(Dense(xdim+c,4,swish), Dense(4, 4, swish), SplitLayer(4, [xdim,xdim], [identity, safe_softplus]))

α = softmax(Float32.(randn(c)))             # α is a trainable parameter!
qz_xy = ConditionalMvNormal(net_xz)
pz = MvNormal(zeros(Float32, zdim), 1f0)    # check that eltype is Float32

α_qy_x = Chain(Dense(2,2,swish), Dense(2,c),softmax)
#qy_x(x) = Categorical(α_qy_x(x))
qy_x = ConditionalCategorical(α_qy_x)

px_yz = ConditionalMvNormal(net_zx)

ps = Flux.params(α, qz_xy, qy_x, px_yz)
opt = ADAM()

# start testing loss functions
include(scriptsdir("conditional_losses.jl"))
x = randn(Float32, 2, 10)
y = sample(1:c, 10)
loss_known(x, y)
loss_unknown(x)
loss_classification(x, y)

function semisupervised_loss(xk, y, xu, N)
    # known and unknown losses
    l_known = loss_known(xk, y)
    l_unknown = loss_unknown(xu)

    # classification loss on known data
    lc = 0.1 * N * loss_classification(xk, y)

    return l_known + l_unknown + lc
end

# load data
ratios = (0.2, 0.4, 0.4)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_data(Float32.(X), y; ratios=ratios)

loss_known(Xk, yk)
loss_unknown(Xk)
loss_unknown(Xu)
loss_classification(Xk, yk)


function accuracy(X, y, c)
    N = length(y)
    ynew = map(xi -> Flux.onecold(condition(qy_x2, xi).p, 1:c), eachcol(X))
    sum(ynew .== y)/N
end
accuracy(X, y) = accuracy(X, y, c)

# only works for Xk and Xu having the same length
# data = Flux.Data.DataLoader((Xk, yk, Xu), batchsize=64)
function minibatch(Xk, y, Xu;ksize=64, usize=64)
    kix = sample(1:size(Xk, 2), ksize)
    uix = sample(1:size(Xu, 2), usize)

    xk, yk = Xk[:, kix], y[kix]
    xu = Xu[:, uix]

    return xk, yk, xu
end
data = minibatch(Xk, yk, Xu)

loss(xk, y, xu) = semisupervised_loss(xk, y, xu, 64)

using Flux: @epochs
@epochs 1000 begin
    Flux.train!(loss, ps, data, opt)
    @show loss(Xk, yk, Xu)
    @show accuracy(Xt, yt)
end

r = probs(condition(qy_x2, Xk))
bar(r[1, :], color=Int.(yk), size=(900,200), ylims=(0,1))


get_rgb(X) = map(xi -> RGBA(condition(qy_x2, xi).p..., 0), eachcol(X))

function discriminative_space()
    Xgrid = []
    for x in -7.5:0.15:7
        for y in -7.3:0.15:3.5
            push!(Xgrid, [x,y])
        end
    end
    Xgrid = hcat(Xgrid...)
    col = get_rgb(Xgrid);
    scatter2(Xgrid, markerstrokewidth=0, markersize=2.8, marker=:square, color=col, opacity=0.5)
end

discriminative_space();
scatter2!(Xk, zcolor=Int.(yk), ms=6);
scatter2!(Xu, zcolor=Int.(yu), marker=:square, opacity=0.7, ms=2.5)
