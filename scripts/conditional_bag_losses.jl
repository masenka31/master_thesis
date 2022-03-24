function loss_known_bag_vec(Xb, y, c)
    Xdata = project_data(Xb)
    n = size(Xdata, 2)
    ydata = repeat([y], n)

    v = bagmodel(Xb)
    vdata = reshape(repeat(v, n), length(v), n)

    # get the concatenation with label (onehot encoding?)
    # yoh = Float32.(Flux.onehotbatch(ydata, 1:c))
    yoh = Flux.onehotbatch(ydata, 1:c) .* ones(Float32, c, n)
    xy = vcat(Xdata, yoh, vdata)

    # get the conditional and sample latent
    qz = condition(qz_xy, xy)
    z = rand(qz) # no need to do the reparametrization trick thanks to DistributionsAD
    yz = vcat(z, yoh)

    # E log p(x|y,z)
    l1 = -logpdf(px_yz, Xdata, yz)

    # log p(y)
    l2 = -logpdf(Categorical(softmax(α)), y)

    # KL divergence on latent
    # l3 = sum(_kld_gaussian(qz_xy(xy), pz))
    l3 = kl_divergence(condition(qz_xy, xy), pz)
    return l1 .+ l3' .+ l2
end
loss_known_bag_vec(Xb, y) = loss_known_bag_vec(Xb, y, c)

function loss_known_bag_vec_onehot(Xb, yohfloat32)
    Xdata = project_data(Xb)
    n = size(Xdata, 2)

    v = bagmodel(Xb)
    vdata = reshape(repeat(v, n), length(v), n)
    # concatenate
    xy = vcat(Xdata, yohfloat32, vdata)

    # get the conditional and sample latent
    qz = condition(qz_xy, xy)
    z = rand(qz) # no need to do the reparametrization trick thanks to DistributionsAD
    yz = vcat(z, yohfloat32)

    # E log p(x|y,z)
    l1 = -logpdf(px_yz, Xdata, yz)

    # log p(y)
    l2 = -logpdf(Categorical(softmax(α)), y)

    # KL divergence on latent
    # l3 = sum(_kld_gaussian(qz_xy(xy), pz))
    l3 = kl_divergence(condition(qz_xy, xy), pz)
    return l1 .+ l3' .+ l2
end

# which one to use?
loss_known_bag(Xb, y) = sum(loss_known_bag_vec(Xb, y))
loss_known_bag(Xb, y) = mean(loss_known_bag_vec(Xb, y))


# """
#     loss_known_bag(Xb, y)

# Calculates the loss for every instance in a bag and returns the sum.
# """
# function loss_known_bag(Xb, y)
#     Xdata = project_data(Xb)
#     ydata = repeat([y], size(Xdata, 2))
#     sum(loss_known_vec(vcat(Xdata, vdata), ydata))
# end


"""
Currently not working.

Should take multiple bags and return the mean bag loss.
"""
function loss_known_mill(Xb, y)
    n = nobs(Xb)
    mean([loss_known_bag(Xb[i], y[i]) for i in 1:n])
    #mapreduce(i -> loss_known_bag(Xb[i], y[i]), mean, 1:n)
end

function loss_unknown(Xb, c) # x::AbstractMatrix
    lmat = mapreduce(y -> loss_known_bag_vec(Xb, y), hcat, 1:c)
    # lmat = mapreduce(y -> loss_known_vec(vcat(Xdata, vdata), repeat([y], n)), hcat, 1:c)
    # x_α = bagmodel(Xb)
    # prob = condition(qy_x, x_α).α
    prob = condition(qy_x, bagmodel(Xb)).α

    # l = mean(sum(lmat' .* prob, dims=1))
    l = sum(lmat' .* prob)
    # e = - mean(entropy(condition(qy_x, x_α)))
    # minus or plus? don't we want to minimize entropy rather than maximize it?
    # n = size(project_data(Xb), 2)
    e = - mean(entropy(condition(qy_x, bagmodel(Xb))))
    return l + e
end
loss_unknown(Xb) = loss_unknown(Xb, c)

# @time yohfullfloat32 = map(c -> Float32.(Flux.onehotbatch(repeat([c], size(project_data(Xb), 2)), 1:10)), 1:10)


function loss_unknown_onehot(Xb)
    yohfullfloat32 = map(c -> Float32.(Flux.onehotbatch(repeat([c], size(project_data(Xb), 2)), 1:10)), 1:10)
    lmat = mapreduce(y -> loss_known_bag_vec_onehot(Xb, y), hcat, yohfullfloat32)
    prob = condition(qy_x, bagmodel(Xb)).α

    l = mean(sum(lmat' .* prob, dims=1))
    # e = - mean(entropy(condition(qy_x, x_α)))
    # minus or plus? don't we want to minimize entropy rather than maximize it?
    e = - mean(entropy(condition(qy_x, bagmodel(Xb))))
    return l - e
end

loss_unknown_mill(Xb, c) = mean([loss_unknown(Xb[i], c) for i in 1:nobs(Xb)])
loss_unknown_mill(Xb) = loss_unknown_mill(Xb, c)

function loss_classification(Xb, y)
    #- logpdf(condition(qy_x, bagmodel(Xb)[:]), y)
    - mean(logpdf(condition(qy_x, bagmodel(Xb)), [y]))
end

function loss_classification_crossentropy(Xb, y)
    #- logpdf(condition(qy_x, bagmodel(Xb)[:]), y)
    yoh = Flux.onehot(y, 1:c)
    Flux.binarycrossentropy(condition(qy_x, bagmodel(Xb)[:]).p, yoh)
end

loss_classification_mill(Xb, y) = mean([loss_classification(Xb[i], y[i]) for i in 1:nobs(Xb)])


function reconstruct_mean(Xb, y)
    Xdata = project_data(Xb)
    n = size(Xdata, 2)
    ydata = repeat([y], n)

    v = bagmodel(Xb)
    vdata = reshape(repeat(v, n), length(v), n)

    # get the concatenation with label (onehot encoding?)
    yoh = Flux.onehotbatch(ydata, 1:c)
    xy = vcat(Xdata, yoh, vdata)

    # get the conditional and sample latent
    qz = condition(qz_xy, xy)
    z = rand(qz)
    z = mean(qz)
    yz = vcat(z, yoh)

    mean(condition(px_yz, yz))
end
function reconstruct_rand(Xb, y)
    Xdata = project_data(Xb)
    n = size(Xdata, 2)
    ydata = repeat([y], n)

    v = bagmodel(Xb)
    vdata = reshape(repeat(v, n), length(v), n)

    # get the concatenation with label (onehot encoding?)
    yoh = Flux.onehotbatch(ydata, 1:c)
    xy = vcat(Xdata, yoh, vdata)

    # get the conditional and sample latent
    qz = condition(qz_xy, xy)
    z = rand(qz)
    z = mean(qz)
    yz = vcat(z, yoh)

    rand(condition(px_yz, yz))
end
