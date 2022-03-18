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

function split_semisupervised_data(X::AbstractMillNode, y::Vector; ratios=(0.3,0.3,0.4), seed=nothing)
    
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    n = length(y)
    ix = sample(1:n, n, replace=false)
    nk, nu, nt = round.(Int, n .* ratios)

    Xk, Xu, Xt = X[ix[1:nk]], X[ix[nk+1:nk+nu]], X[ix[nk+nu+1:n]]
    yk, yu, yt = y[ix[1:nk]], y[ix[nk+1:nk+nu]], y[ix[nk+nu+1:n]]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return Xk, yk, Xu, yu, Xt, yt
end