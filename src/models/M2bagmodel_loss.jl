"""
    encoder(model<:M2BagModel, Xb, y)

Returns the projection of a bag to the encoding used for q(z | ...)
based on the model type. Also returns the onehot encoding for each
instance based on the label.
"""
function encoder(model::M2Bag, Xb, y, c)
    Xdata = project_data(Xb)
    n = size(Xdata, 2)
    ydata = repeat([y], n)

    # get the concatenation with label (onehot encoding?)
    yoh = Flux.onehotbatch(ydata, 1:c) .* ones(Float32, c, n)
    xy = vcat(Xdata, yoh)
    return Xdata, xy, yoh
end
function encoder(model::M2BagDense, Xb, y, c)
    Xdata = project_data(Xb)
    n = size(Xdata, 2)
    ydata = repeat([y], n)
    bvec = model.bagmodel(Xb)
    bdata = repeat(bvec, outer=(1, n))

    # get the concatenation with label (onehot encoding?)
    yoh = Flux.onehotbatch(ydata, 1:c) .* ones(Float32, c, n)
    xby = vcat(Xdata, bdata, yoh)
    return Xdata, xby, yoh
end
function encoder(model::M2BagSimple, Xb, y, c)
    Xdata = project_data(Xb)
    n = size(Xdata, 2)
    ydata = repeat([y], n)
    yoh = Flux.onehotbatch(ydata, 1:c) .* ones(Float32, c, n)
    return Xdata, Xdata, yoh
end

function loss_known_bag_vec(model::M2BagModel, Xb, y, c)
    # get the input to q(z | ...) and onehot encoding
    # dispatched on model types
    Xdata, enc, yoh = encoder(model, Xb, y, c)

    # get the conditional and sample latent
    qz = condition(model.qz_xy, enc)
    z = rand(qz) # no need to do the reparametrization trick thanks to DistributionsAD
    yz = vcat(z, yoh)

    # E log p(x|y,z)
    l1 = -logpdf(model.px_yz, Xdata, yz)

    # log p(y)
    l2 = -logpdf(Categorical(softmax(model.α)), y)

    # KL divergence on latent
    # l3 = sum(_kld_gaussian(qz_xy(xy), pz))
    l3 = kl_divergence(condition(model.qz_xy, enc), model.prior)
    return l1 .+ l3' .+ l2
end

# which one to use?
loss_known_bag(model::M2BagModel, Xb, y, c) = sum(loss_known_bag_vec(model, Xb, y, c))
# loss_known_bag(model::M2BagModel, Xb, y) = mean(loss_known_bag_vec(model, Xb, y))

function loss_unknown(model::M2BagModel, Xb, c) # x::AbstractMatrix
    lmat = mapreduce(y -> loss_known_bag_vec(model, Xb, y, c), hcat, 1:c)
    # prob = condition(model.qy_x, model.bagmodel(Xb)).α
    prob = softmax(model.bagmodel(Xb))

    l = sum(lmat' .* prob)
    # e = - mean(entropy(condition(model.qy_x, model.bagmodel(Xb))))
    e = - entropy(prob)
    return l + e
end

function loss_classification(model::M2BagModel, Xb, y::Int)
    #- logpdf(condition(qy_x, bagmodel(Xb)[:]), y)
    - mean(logpdf(condition(model.qy_x, model.bagmodel(Xb)), [y]))
end
function loss_classification(model::M2BagModel, Xb, y::Vector{T}) where T<:Int
    #- logpdf(condition(qy_x, bagmodel(Xb)[:]), y)
    - mean(logpdf(condition(model.qy_x, model.bagmodel(Xb)), y))
end

function loss_classification_crossentropy(model::M2BagModel, Xb, y, c)
    #- logpdf(condition(qy_x, bagmodel(Xb)[:]), y)
    # Flux.crossentropy(condition(model.qy_x, model.bagmodel(Xb)).α, Flux.onehotbatch(y, 1:c))
    Flux.logitcrossentropy(model.bagmodel(Xb), Flux.onehotbatch(y, 1:c))
end

function semisupervised_loss(model::M2BagModel, xk, y, xu, c, N)
    # known and unknown losses
    l_known = loss_known_bag(model, xk, y, c)
    l_unknown = loss_unknown(model, xu, c)

    # classification loss on known data
    # lc = N * loss_classification(model, xk, y)
    lc = N * loss_classification_crossentropy(model, xk, y, c)

    return l_known + l_unknown + lc
end

### Try Chamfer distance ###

function loss_known_bag_vec_Chamfer(model::M2BagModel, Xb, y, c)
    # get the input to q(z | ...) and onehot encoding
    # dispatched on model types
    Xdata, enc, yoh = encoder(model, Xb, y, c)

    # get the conditional and sample latent
    qz = condition(model.qz_xy, enc)
    z = rand(qz) # no need to do the reparametrization trick thanks to DistributionsAD
    yz = vcat(z, yoh)

    # E log p(x|y,z)
    # l1 = -logpdf(model.px_yz, Xdata, yz)

    # try Chamfer distance
    xhat = rand(condition(model.px_yz, yz))
    l1 = chamfer_distance(xhat, Xdata)

    # log p(y)
    l2 = -logpdf(Categorical(softmax(model.α)), y)

    # KL divergence on latent
    # l3 = sum(_kld_gaussian(qz_xy(xy), pz))
    l3 = kl_divergence(condition(model.qz_xy, enc), model.prior)
    return l1 .+ l3' .+ l2
end

loss_known_bag_Chamfer(model::M2BagModel, Xb, y, c) = sum(loss_known_bag_vec_Chamfer(model, Xb, y, c))

function loss_unknown_Chamfer(model::M2BagModel, Xb, c) # x::AbstractMatrix
    lmat = mapreduce(y -> loss_known_bag_vec_Chamfer(model, Xb, y, c), hcat, 1:c)
    # prob = condition(model.qy_x, model.bagmodel(Xb)).α
    prob = softmax(model.bagmodel(Xb))

    l = sum(lmat' .* prob)
    # e = - mean(entropy(condition(model.qy_x, model.bagmodel(Xb))))
    e = entropy(prob)
    return l + e
end

function semisupervised_loss_Chamfer(model::M2BagModel, xk, y, xu, c, N)
    # known and unknown losses
    l_known = loss_known_bag_Chamfer(model, xk, y, c)
    l_unknown = loss_unknown_Chamfer(model, xu, c)

    # classification loss on known data
    # lc = N * loss_classification(model, xk, y)
    lc = N * loss_classification_crossentropy(model, xk, y, c)

    return l_known + l_unknown + lc
end

#######################
### Reconstructions ###
#######################

function reconstruct(model::M2BagModel, Xb, y, c)
    _, enc, yoh = encoder(model, Xb, y, c)
    qz = condition(model.qz_xy, enc)
    z = rand(qz)
    yz = vcat(z, yoh)
    rand(condition(model.px_yz, yz))
end

function reconstruct_mean(model::M2BagModel, Xb, y, c)
    _, enc, yoh = encoder(model, Xb, y, c)
    qz = condition(model.qz_xy, enc)
    z = rand(qz)
    yz = vcat(z, yoh)
    condition(model.px_yz, yz).μ
end

function chamfer_score(model::M2BagModel, Xb, classes)
    c = length(classes)
    scores = zeros(c)
    for y in 1:c
        Xhat = reconstruct(model, Xb, y, c)
        scores[y] = chamfer_distance(project_data(Xb), Xhat)
    end
    classes[findmin(scores)[2]]
end

#################################################
### Other losses for "effective" minibatching ###
#################################################

lknown(xk, y) = loss_known_bag_Chamfer(model, xk, y, c)
lunknown(xu) = loss_unknown_Chamfer(model, xu, c)

# reconstruction loss - known + unknown
function loss_rec(Xk, yk, Xu)
    l_known = mean(lknown.(Xk, yk))
    l_unknown = mean(lunknown.(Xu))
    return l_known + l_unknown
end

# this needs to be in a script
# N = size(project_data(Xk), 2)
# lclass(x, y) = loss_classification_crossentropy(model, x, y, c) * 0.1f0 * N

# # now we should be able to dispatch over bags and labels
# function lossf(Xk, yk, Xu)
#     nk = nobs(Xk)
#     bk = Flux.Zygote.@ignore [Xk[i] for i in 1:nk]

#     nu = nobs(Xu)
#     bu = Flux.Zygote.@ignore [Xu[i] for i in 1:nu]
    
#     lr = loss_rec(bk, yk, bu)
#     lc = lclass(Xk, yk)
#     return lr + lc
# end
