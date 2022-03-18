function loss_known(Xb, y)
    X = project_data(Xb)
    # get the concatenation with label (onehot encoding?)
    yoh = Flux.onehot(y, 1:c)
    xy = [vcat(x, yoh) for x in eachcol(X)]

    # get the conditional and sample latent
    qz = [qz_xy(xyi) for xyi in xy]
    z = rand.(qz) # no need to do the reparametrization trick thanks to DistributionsAD
    yz = [vcat(zi, yoh) for zi in z]

    # E log p(x|y,z)
    l1 = -sum([logpdf(px_yz(yzi), xi) for (yzi, xi) in zip(yz, eachcol(X))])

    # log p(y)
    l2 = -logpdf(Categorical(softmax(α)), y)

    # KL divergence on latent
    l3 = sum(sum([_kld_gaussian(qz_xy(xyi), pz) for xyi in xy]))

    return l1 + l2 + l3 
end

# this stays the same as well as the classification loss
function loss_unknown(x, c)
    l = sum([loss_known(x, y) for y in c] .* pdf.(qy_x(x), c))
    e = - entropy(qy_x(x))
    return l + e
end
loss_unknown_sumx(x) = loss_unknown(x, collect(1:c))