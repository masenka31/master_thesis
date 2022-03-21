function loss_known(x, y)
    # get the concatenation with label (onehot encoding?)
    yoh = Flux.onehot(y, 1:c)
    xy = vcat(x, yoh)

    # get the conditional and sample latent
    qz = qz_xy(xy)
    z = rand(qz) # no need to do the reparametrization trick thanks to DistributionsAD
    yz = vcat(z, yoh)

    # E log p(x|y,z)
    l1 = -logpdf(px_yz(yz), x)

    # log p(y)
    l2 = -logpdf(Categorical(softmax(α)), y)

    # KL divergence on latent
    l3 = sum(_kld_gaussian(qz_xy(xy), pz))

    return l1 + l2 + l3 
end

function loss_known2(x, y)
    # get the concatenation with label (onehot encoding?)
    yoh = Flux.onehot(y, 1:c)
    xy = vcat(x, yoh)

    # get the conditional and sample latent
    qz = qz_x(x)
    z = rand(qz) # no need to do the reparametrization trick thanks to DistributionsAD
    yz = vcat(z, yoh)

    # E log p(x|y,z)
    l1 = -logpdf(px_yz(yz), x)

    # log p(y)
    l2 = -logpdf(Categorical(softmax(α)), y)

    # KL divergence on latent
    l3 = sum(_kld_gaussian(qz_x(x), pz))

    return l1 + l2 + l3 
end

function loss_unknown(x, c)
    l = sum([loss_known(x, y) for y in c] .* pdf.(qy_x(x), c))
    e = - entropy(qy_x(x))
    return l + e
end
loss_unknown_sumx(x) = loss_unknown(x, collect(1:c))

function loss_classification(x, y)
    - logpdf(qy_x(x), y)
end

#function latent_rep(x, y, c)
