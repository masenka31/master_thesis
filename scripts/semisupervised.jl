# semisupervised VAE

using DrWatson
using master_thesis
using Distributions, DistributionsAD
using Flux

using StatsBase
using LinearAlgebra
using Distances
using Base.Iterators: repeated

using Plots
gr(label="");

##########################################################
################## semisupervised model ##################
##########################################################

include(scriptsdir("losses.jl"))

function minibatch(Xk, y, Xu;ksize=64, usize=64)
    kix = sample(1:size(Xk, 2), ksize)
    uix = sample(1:size(Xu, 2), usize)

    xk, yk = [Xk[:, i] for i in kix], y[kix]
    xu = [Xu[:, i] for i in uix]

    return xk, yk, xu
end

function semisupervised_loss(xk, y, xu, N)
    # known and unknown losses
    l_known = loss_known(xk, y)
    l_unknown = loss_unknown_sumx(xu)

    # classification loss on known data
    lc = 0.1 * N * loss_classification(xk, y)

    return l_known + l_unknown + lc
end

function accuracy(X, y, c)
    N = length(y)
    ynew = map(xi -> Flux.onecold(qy_x(xi).p, 1:c), eachcol(X))
    sum(ynew .== y)/N
end
accuracy(X, y) = accuracy(X, y, c)

# initialize data and models
ratios = (0.05, 0.45, 0.5)
include(scriptsdir("init.jl"))
ksize, usize = round.(Int, 128 .* ratios[1:2])
ksize, usize = 64, 64
loss(xk, yk, xu) = semisupervised_loss(xk, yk, xu, ksize)
# loss(xk, yk, xu) = semisupervised_loss(xk, yk, xu, 32)

using Flux: @epochs
anim = @animate for i in 1:200
    space_plot()
    b = minibatch(Xk, yk, Xu; ksize=ksize, usize=usize)
    Flux.train!(loss, ps, zip(b...), opt)
    @show i
    a = round(accuracy(Xt, yt), digits=4)
    @show a
    #if a > 0.98
    #    break
    #end
end
gif(anim, "animation.gif", fps = 15)

accuracy(Xk, yk)
accuracy(Xu, yu)
accuracy(Xt, yt)

# how to visualize the discriminative space?
# there are no boundaries -> we could find them
# or visualize it on a grid based on rgb color

get_rgb(X) = map(xi -> RGBA(qy_x(xi).p..., 0), eachcol(X))

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
scatter2!(Xk, zcolor=Int.(yk), ms=6)
scatter2!(Xu, zcolor=Int.(yu), marker=:square, opacity=0.7, ms=2.5)

function space_plot()
    discriminative_space()
    scatter2!(Xk, zcolor=Int.(yk), ms=6)
    scatter2!(Xu, zcolor=Int.(yu), marker=:square, opacity=0.7, ms=2.5)
end


discriminative_space();
scatter2!(Xt, zcolor=Int.(yt))

r = mapreduce(xi -> qy_x(xi).p, hcat, eachcol(Xt))

bar(r[1,:], color=Int.(yt), size=(1000,400))
bar(r[2,:], color=Int.(yt), size=(1000,400))
bar(r[3,:], color=Int.(yt), size=(1000,400))

# just a discriminator model
classifier = Chain(Dense(2, 5, swish), Dense(5, 5, swish), Dense(5, c))
ps_c = Flux.params(classifier)
loss_c(x, y) = Flux.logitcrossentropy(classifier(x), y)
yoh_c = Flux.onehotbatch(yk, 1:c)
Xk
opt_c = ADAM()

for i in 1:100
    # Flux.train!(loss_c, ps_c, Flux.Data.DataLoader((Xk, yoh_c), batchsize=30), opt_c)
    Flux.train!(loss_c, ps_c, repeated((Xk, yoh_c), 10), opt_c)
    @show loss_c(Xk, yoh_c)
end

get_rgb_classifier(X) = map(xi -> RGBA(softmax(classifier(xi))..., 0), eachcol(X))
function classifier_space()
    Xgrid = []
    for x in -7.5:0.15:7
        for y in -7.3:0.15:3.5
            push!(Xgrid, [x,y])
        end
    end
    Xgrid = hcat(Xgrid...)
    col = get_rgb_classifier(Xgrid);
    scatter2(Xgrid, markerstrokewidth=0, markersize=2.8, marker=:square, color=col, opacity=0.5)
end

classifier_space();
scatter2!(Xk, zcolor=Int.(yk), ms=6);
scatter2!(Xu, zcolor=Int.(yu), marker=:square, opacity=0.7, ms=2.5)

space_plot()