using Random

"""
    random_ix(n::Int, seed=nothing)

This function generates random indexes based on the maximum number
and given seed. If no seed is set, samples randomly.
"""
function random_ix(n::Int, seed=nothing)
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    _ix = sample(1:n, n; replace=false)

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return _ix
end

"""
    train_test_split(X::AbstractMillNode, y; ratio=0.5, seed=nothing)

Classic train/test split with given ratio.
"""
function train_test_split(X::AbstractMillNode, y::Vector; ratio=0.5, seed=nothing)
    n = length(y)
    n1 = round(Int, ratio*n)

    # get the indexes
    _ix = random_ix(n, seed)
    
    # split indexes to train/test
    train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]

    # get data
    Xtrain, ytrain = X[train_ix], y[train_ix]
    Xtest, ytest = X[test_ix], y[test_ix]

    return (Xtrain, ytrain), (Xtest, ytest)
end

"""
	seqids2bags(bagids)

"""
function seqids2bags(bagids)
	c = countmap(bagids)
	Mill.length2bags([c[i] for i in sort(collect(keys(c)))])
end

"""
	reindex(bagnode, inds)

A faster implementation of Base.getindex.
"""
function reindex(bagnode::BagNode, inds)
	obs_inds = bagnode.bags[inds]
	new_bagids = vcat(map(x->repeat([x[1]], length(x[2])), enumerate(obs_inds))...)
	data = bagnode.data.data[:,vcat(obs_inds...)]
	new_bags = seqids2bags(new_bagids)
	BagNode(ArrayNode(data), new_bags)
end
reindex(bagnode::ProductNode, inds) = bagnode[inds]

function split_semisupervised(X::AbstractMillNode, y::Vector; ratios=(0.3,0.3,0.4), seed=nothing)
    
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    n = length(y)
    ix = sample(1:n, n, replace=false)
    nk, nu, nt = round.(Int, n .* ratios)

    Xk, Xu, Xt = reindex(X, ix[1:nk]), reindex(X, ix[nk+1:nk+nu]), reindex(X, ix[nk+nu+1:n])
    yk, yu, yt = y[ix[1:nk]], y[ix[nk+1:nk+nu]], y[ix[nk+nu+1:n]]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return Xk, yk, Xu, yu, Xt, yt
end

"""
    split_semisupervised_balanced(X::AbstractMillNode, y::Vector; ratios=(0.3,0.3,0.4), seed=nothing)

Splits data based on given ratios = (labeled, unlabeled, test).
"""
function split_semisupervised_balanced(X::AbstractMillNode, y::Vector; ratios=(0.3,0.3,0.4), seed=nothing)
    
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    n = length(y)
    un = unique(y)
    c = length(un)

    # create balanced classes
    nk, nu, nt = round.(Int, n .* ratios)
    r = round(Int, nk / c)
    nk = r * c

    ik = []
    for i in 1:c
        avail_ix = (1:n)[y .== un[i]]
        ix = sample(avail_ix, r, replace=false)
        push!(ik, ix)
    end
    ik = shuffle(vcat(ik...))

    ix_left = setdiff(1:n, ik)
    ix = sample(ix_left, length(ix_left), replace=false)      # sample can be fixed by seed :)
    
    iu, it = ix[1:nu], ix[nu+1:end]

    # Xk, Xu, Xt = reindex(X, ix[1:nk]), reindex(X, ix[nk+1:nk+nu]), reindex(X, ix[nk+nu+1:n])
    Xk, Xu, Xt = reindex(X, ik), reindex(X, iu), reindex(X, it)
    # yk, yu, yt = y[ix[1:nk]], y[ix[nk+1:nk+nu]], y[ix[nk+nu+1:n]]
    yk, yu, yt = y[ik], y[iu], y[it]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return Xk, yk, Xu, yu, Xt, yt
end

function validation_catch(avail_ix, n, seed)
    try
        (seed == nothing) ? nothing : Random.seed!(seed)
        return sample(avail_ix, n, replace=false)
        (seed !== nothing) ? Random.seed!() : nothing
    catch e
        return sample(avail_ix, length(avail_ix), replace=false)
    end
end

"""
    validation_data(yk, Xu, yu, seed, classes)

Takes `length(yk)` number of samples from unlabeled dataset
as validation dataset.
"""
function validation_data(yk, Xu, yu, seed, classes)
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    c = length(classes)
    n = round(Int, length(yk) / c)
    N = length(yu)

    ik = []
    for i in 1:c
        avail_ix = (1:N)[yu .== classes[i]]
        ix = validation_catch(avail_ix, n, seed)
        push!(ik, ix)
    end
    ik = shuffle(vcat(ik...))
    ileft = setdiff(1:N, ik)

    x, y = reindex(Xu, ik), yu[ik]
    new_xu, new_yu = reindex(Xu, ileft), yu[ileft]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return x, y, new_xu, new_yu
end