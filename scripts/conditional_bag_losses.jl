"""
    loss_known_bag(Xb, y)

Calculates the loss for every instance in a bag and returns the sum.
"""
function loss_known_bag(Xb, y)
    Xdata = project_data(Xb)
    ydata = repeat([y], size(Xdata, 2))
    sum(loss_known_vec(Xdata, ydata))
end

function reconstruct_mean(Xb, y)
    Xdata = project_data(Xb)
    ydata = repeat([y], size(Xdata, 2))

    yoh = Flux.onehotbatch(ydata, 1:c)
    xy = vcat(Xdata, yoh)

    qz = condition(qz_xy, xy)
    z = rand(qz)
    yz = vcat(z, yoh)

    mean(condition(px_yz, yz))
end
function reconstruct_rand(Xb, y)
    Xdata = project_data(Xb)
    ydata = repeat([y], size(Xdata, 2))

    yoh = Flux.onehotbatch(ydata, 1:c)
    xy = vcat(Xdata, yoh)

    qz = condition(qz_xy, xy)
    z = rand(qz)
    yz = vcat(z, yoh)

    rand(condition(px_yz, yz))
end


#Base.length(x::BagNode) = nobs(x)
function loss_known_mill(Xb, y)
    n = nobs(Xb)
    mean([loss_known_bag(Xb[i], y[i]) for i in 1:n])
    #mapreduce(i -> loss_known_bag(Xb[i], y[i]), mean, 1:n)
end

function loss_unknown(Xb, c) # x::AbstractMatrix
    Xdata = project_data(Xb)
    n = size(Xdata, 2)

    lmat = mapreduce(y -> loss_known_vec(Xdata, repeat([y], n)), hcat, 1:c)
    # x_α = bagmodel(Xb)
    # prob = condition(qy_x, x_α).α
    prob = condition(qy_x, bagmodel(Xb)).α

    l = mean(sum(lmat' .* prob, dims=1))
    # e = - mean(entropy(condition(qy_x, x_α)))
    e = - mean(entropy(condition(qy_x, bagmodel(Xb))))
    return l + e
end
loss_unknown(Xb) = loss_unknown(Xb, c)

loss_unknown_mill(Xb, c) = mean([loss_unknown(Xb[i], c) for i in 1:nobs(Xb)])
loss_unknown_mill(Xb) = loss_unknown_mill(Xb, c)

function loss_classification(Xb, y)
    #- logpdf(condition(qy_x, bagmodel(Xb)[:]), y)
    - mean(logpdf(condition(qy_x, bagmodel(Xb)), [y]))
end
loss_classification_mill(Xb, y) = mean([loss_classification(Xb[i], y[i]) for i in 1:nobs(Xb)])
