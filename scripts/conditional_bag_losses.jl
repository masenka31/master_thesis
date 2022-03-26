using Flux.Zygote: @ignore

repeat_y(y, n, y_lengths) = mapreduce(i -> repeat([y[i]], y_lengths[i]), vcat, 1:n)

"""
    loss_known_bag(Xb, yb)

For all instances in a given batch of BagNodes and vector of labels calculates
the average point loss for data with known labels.

Works for one bag and an Integer label as well.
"""
function loss_known_bag(Xb, yb)
    # this is just data manipulation
    # Flux.Zygote.ignore() do
    #     nbags = nobs(Xb)
    #     y_lengths = length.(Xb.bags)
    #     y_i = mapreduce(i -> repeat([yb[i]], y_lengths[i]), vcat, 1:nbags)
    # end

    # y_i = mapreduce(i -> repeat([yb[i]], y_lengths[i]), vcat, 1:nbags)
    y_i = Flux.Zygote.ignore() do
        nbags = nobs(Xb)
        y_lengths = length.(Xb.bags)
        repeat_y(yb, nbags, y_lengths)
    end

    # this is what needs to be differentiated
    loss_known(project_data(Xb), y_i)
end

# this only works for one bag
# function loss_unknown_bag(Xb)
#     lknown = map(y -> loss_known_bag(Xb, y), 1:c)
#     prob = condition(qy_x, bagmodel(Xb)).α
#     l = dot(lknown, prob)

#     return l - mean(entropy(condition(qy_x, bagmodel(Xb))))
# end

"""
    loss_unknown_bag(Xb)

Works both for one and more bags. Should be able to iterate over bags and labels
and return the desired loss function for data with unknown labels.
"""
function loss_unknown_bag(Xb)
    nbags = nobs(Xb)
    bagvec = @ignore [Xb[i] for i in 1:nbags]
    L = mapreduce(y -> map(x -> loss_known_bag(x, y), bagvec), hcat, 1:c)
    prob = condition(qy_x, bagmodel(Xb)).α
    Lmat = prob * L
    # l1 = mean(sum(Lmat, dims=1))
    l1 = mean(Lmat)

    l2 = - mean(entropy(condition(qy_x, bagmodel(Xb))))
    return l1 + l2
end

function loss_classification(x, y)
    #- mean(logpdf(qy_x, y, x))
    - mean(logpdf(condition(qy_x, bagmodel(x)), y))
end


######################################################################
# OLD 

function loss_known_bag_vec(Xb, y)
    Xdata = project_data(Xb)
    n = size(Xdata, 2)
    ydata = repeat([y], n)

    # get the concatenation with label (onehot encoding?)
    # yoh = Float32.(Flux.onehotbatch(ydata, 1:c))
    yoh = Flux.onehotbatch(ydata, 1:c) .* ones(Float32, c, n)
    xy = vcat(Xdata, yoh)

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

# which one to use?
loss_known_bag(Xb, y) = sum(loss_known_bag_vec(Xb, y))
loss_known_bag(Xb, y) = mean(loss_known_bag_vec(Xb, y))


# # """
# #     loss_known_bag(Xb, y)

# # Calculates the loss for every instance in a bag and returns the sum.
# # """
# # function loss_known_bag(Xb, y)
# #     Xdata = project_data(Xb)
# #     ydata = repeat([y], size(Xdata, 2))
# #     sum(loss_known_vec(vcat(Xdata, vdata), ydata))
# # end


# """
# Currently not working.

# Should take multiple bags and return the mean bag loss.
# """
# function loss_known_mill(Xb, y)
#     n = nobs(Xb)
#     mean([loss_known_bag(Xb[i], y[i]) for i in 1:n])
#     #mapreduce(i -> loss_known_bag(Xb[i], y[i]), mean, 1:n)
# end

function loss_unknown(Xb) # x::AbstractMatrix
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
# loss_unknown(Xb) = loss_unknown(Xb, c)

# # @time yohfullfloat32 = map(c -> Float32.(Flux.onehotbatch(repeat([c], size(project_data(Xb), 2)), 1:10)), 1:10)


# function loss_unknown_onehot(Xb)
#     yohfullfloat32 = map(c -> Float32.(Flux.onehotbatch(repeat([c], size(project_data(Xb), 2)), 1:10)), 1:10)
#     lmat = mapreduce(y -> loss_known_bag_vec_onehot(Xb, y), hcat, yohfullfloat32)
#     prob = condition(qy_x, bagmodel(Xb)).α

#     l = mean(sum(lmat' .* prob, dims=1))
#     # e = - mean(entropy(condition(qy_x, x_α)))
#     # minus or plus? don't we want to minimize entropy rather than maximize it?
#     e = - mean(entropy(condition(qy_x, bagmodel(Xb))))
#     return l - e
# end

# loss_unknown_mill(Xb, c) = mean([loss_unknown(Xb[i], c) for i in 1:nobs(Xb)])
# loss_unknown_mill(Xb) = loss_unknown_mill(Xb, c)

function loss_classification(Xb, y)
    #- logpdf(condition(qy_x, bagmodel(Xb)[:]), y)
    - mean(logpdf(condition(qy_x, bagmodel(Xb)), [y]))
end

# function loss_classification_crossentropy(Xb, y)
#     #- logpdf(condition(qy_x, bagmodel(Xb)[:]), y)
#     yoh = Flux.onehot(y, 1:c)
#     Flux.binarycrossentropy(condition(qy_x, bagmodel(Xb)[:]).p, yoh)
# end

# loss_classification_mill(Xb, y) = mean([loss_classification(Xb[i], y[i]) for i in 1:nobs(Xb)])


# function reconstruct_mean(Xb, y)
#     Xdata = project_data(Xb)
#     n = size(Xdata, 2)
#     ydata = repeat([y], n)

#     v = bagmodel(Xb)
#     vdata = reshape(repeat(v, n), length(v), n)

#     # get the concatenation with label (onehot encoding?)
#     yoh = Flux.onehotbatch(ydata, 1:c)
#     xy = vcat(Xdata, yoh, vdata)

#     # get the conditional and sample latent
#     qz = condition(qz_xy, xy)
#     z = rand(qz)
#     z = mean(qz)
#     yz = vcat(z, yoh)

#     mean(condition(px_yz, yz))
# end
# function reconstruct_rand(Xb, y)
#     Xdata = project_data(Xb)
#     n = size(Xdata, 2)
#     ydata = repeat([y], n)

#     v = bagmodel(Xb)
#     vdata = reshape(repeat(v, n), length(v), n)

#     # get the concatenation with label (onehot encoding?)
#     yoh = Flux.onehotbatch(ydata, 1:c)
#     xy = vcat(Xdata, yoh, vdata)

#     # get the conditional and sample latent
#     qz = condition(qz_xy, xy)
#     z = rand(qz)
#     z = mean(qz)
#     yz = vcat(z, yoh)

#     rand(condition(px_yz, yz))
# end
