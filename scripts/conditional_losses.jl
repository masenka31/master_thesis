# function loss_known(x, y)
#     # get the concatenation with label (onehot encoding?)
#     yoh = Flux.onehotbatch(y, 1:c)
#     xy = vcat(x, yoh)

#     # get the conditional and sample latent
#     qz = condition(qz_xy, xy)
#     z = rand(qz) # no need to do the reparametrization trick thanks to DistributionsAD
#     yz = vcat(z, yoh)

#     # E log p(x|y,z)
#     l1 = -mean(logpdf(px_yz, x, yz))

#     # log p(y)
#     l2 = -mean(logpdf.(Categorical(softmax(α)), y))

#     # KL divergence on latent
#     # l3 = sum(_kld_gaussian(qz_xy(xy), pz))
#     l3 = mean(kl_divergence(condition(qz_xy, xy), pz))

#     return l1 + l2 + l3 
# end

function loss_known_vec(x, y)
    # get the concatenation with label (onehot encoding?)
    yoh = Flux.onehotbatch(y, 1:c)
    xy = vcat(x, yoh)

    # get the conditional and sample latent
    qz = condition(qz_xy, xy)
    z = rand(qz) # no need to do the reparametrization trick thanks to DistributionsAD
    yz = vcat(z, yoh)

    # E log p(x|y,z)
    l1 = -logpdf(px_yz, x, yz)

    # log p(y)
    l2 = -logpdf.(Categorical(softmax(α)), y)

    # KL divergence on latent
    # l3 = sum(_kld_gaussian(qz_xy(xy), pz))
    l3 = kl_divergence(condition(qz_xy, xy), pz)
    return l1 .+ l2 .+ l3'
end
loss_known(x, y) = mean(loss_known_vec(x, y))

function loss_unknown(x, c) # x::AbstractMatrix
    n = size(x, 2)

    lmat = mapreduce(y -> loss_known_vec(x, repeat([y], n)), hcat, 1:c)
    prob = condition(qy_x, x).α

    l = mean(sum(lmat' .* prob, dims=1))
    e = - mean(entropy(condition(qy_x, x)))
    return l + e
end
# function loss_unknown(x::AbstractVector, c)
#     lx = map(y -> loss_known(x, y), 1:c)
#     prob = probs(condition(qy_x, x))

#     l = dot(lx, prob)
#     e = - mean(entropy(condition(qy_x, x)))
#     return l + e
# end
loss_unknown(x) = loss_unknown(x, c)

function loss_classification(x, y)
    #- mean(logpdf(qy_x, y, x))
    - mean(logpdf(condition(qy_x, x), y))
end

# this is loss for the whole batch on label 1
# loss_known_vec(x, repeat([1], 10))
# I want to multiply it with probability q(1)