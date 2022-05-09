"""
1. Load the data based on the parameters given: r, full.
2. Load the best model.
3. Create a function `embedding(X)`.
4. Use prepared functions to calculate clusterings, umaps etc.
"""

using Clustering, Distances
# using ProgressMeter

"""
    map_clustering(enc_train, enc_test, ytrain, assign; metric=:mse)

Takes encoding of train data and calculates the means of the classes.
For each cluster in test data, finds the closest mean of known data
and assigns the cluster that label.
"""
function map_clustering(enc_train, enc_test, ytrain, assign)
    classes = sort(unique(ytrain))
    train_means = map(i -> mean(enc_train[:, ytrain .== i], dims=2), classes)

    ynew = similar(assign)

    for i in sort(unique(assign))
        x = enc_test[:, assign .== i]
        m = mean(x, dims=2)
        min_ix = findmin(x -> Flux.mse(m, x), train_means)[2]
        ynew[assign .== i] .= classes[min_ix]
    end

    return ynew
end

"""
    map_clustering_advanced(enc_train, enc_test, ytrain, assign, k=3)

Takes encoding of train data, clusters it hierarchicaly into k clusters
and calculates the means of the classes. For each cluster in test data,
finds the closest mean of known data and assigns the cluster that label.
"""
function map_clustering_advanced(enc_train, enc_test, ytrain, assign, k=3)
    classes = sort(unique(ytrain))

    means = []
    labels = []
    for c in classes
        e = enc_train[:, ytrain .== c]
        d = pairwise(Euclidean(), e)
        h = hclust(d, linkage=:average)
        yn = cutree(h, k=k)
        un = unique(yn)
        m = map(i -> mean(e[:, yn .== i], dims=2), un)
        push!(means, m)
        push!(labels, repeat([c], k))
    end

    means = vcat(means...)
    labels = vcat(labels...)

    ynew = similar(assign)

    for i in sort(unique(assign))
        x = enc_test[:, assign .== i]
        m = mean(x, dims=2)
        min_ix = findmin(x -> Flux.mse(m, x), means)[2]
        ynew[assign .== i] .= labels[min_ix]
    end

    return ynew
end

# what about for hierarchical clustering?
hierarchical_single(DM, k) = cutree(hclust(DM, linkage=:single), k=k)
hierarchical_average(DM, k) = cutree(hclust(DM, linkage=:average), k=k)

# clustering accuracy
accuracy(y1::AbstractVector, y2::AbstractVector) = mean(y1 .== y2)

"""
    cluster(DM::AbstractMatrix, y::AbstractVector, clusterfun, k::Int)

Clusters data from given distance matrix and calculates mean silhouette values, RandIndex,
adjusted RandIndex and Mutual information based on true labels `y`.

Returns a DataFrame with method used, k, and the calculated metrics.

The `clusterfun` provided can be
- `kmeans`
- `kmedoids`
- `hierarchical_average`
"""
function cluster(DM::AbstractMatrix, y::AbstractVector, clusterfun, k::Int)
    cbest = clusterfun(DM, k)
    max_silh = mean(silhouettes(cbest, DM))
    ri = randindex(y, cbest)
    mi = mutualinfo(y, cbest)
    return DataFrame(
        :method => "$clusterfun",
        :k => k,
        :randindex => ri[2],
        :adj_randindex => ri[1],
        :MI => mi,
        :silh => max_silh
    ), cbest
end

"""
    cluster(enc, enc_test, y, yt, clusterfun, k, seed; advanced=true, type="")
    cluster(enc, enc_test, DM, y, yt, clusterfun, k, seed; advanced=true, type="")

Clusters data and assigns the clusters labels from the known classes.
Returns a dataframe of calculated metrics.
"""
function cluster(enc, enc_test, y, yt, clusterfun, k, seed; advanced=true, type="")
    advanced ? mapfun = map_clustering_advanced : mapfun = map_clustering

    DM = pairwise(Euclidean(), enc_test)
    df, c = cluster(DM, yt, clusterfun, k)
    if clusterfun in [kmedoids, kmeans]
        ynew = mapfun(enc, enc_test, y, assignments(c))
    else
        ynew = mapfun(enc, enc_test, y, c)
    end
    a = accuracy(yt, ynew)
    da = DataFrame(
        :accuracy => a,
        :new_randindex => randindex(yt, ynew)[2],
        :new_adj_randindex => randindex(yt, ynew)[1],
        :new_MI => mutualinfo(yt, ynew),
        :seed => seed,
        :type => type,
    )
    return hcat(df, da), ynew, c
end
function cluster(enc, enc_test, DM, y, yt, clusterfun, k, seed; advanced=true, type="")
    advanced ? mapfun = map_clustering_advanced : mapfun = map_clustering
    
    df, c = cluster(DM, yt, clusterfun, k)
    if clusterfun in [kmedoids, kmeans]
        ynew = mapfun(enc, enc_test, y, assignments(c))
    else
        ynew = mapfun(enc, enc_test, y, c)
    end
    a = accuracy(yt, ynew)
    da = DataFrame(
        :accuracy => a,
        :new_randindex => randindex(yt, ynew)[2],
        :new_adj_randindex => randindex(yt, ynew)[1],
        :new_MI => mutualinfo(yt, ynew),
        :seed => seed,
        :type => type,
    )
    return hcat(df, da), ynew, c
end

# plotting functions
const colorvec = [:blue4, :green4, :darkorange, :purple3, :red3, :grey, :sienna4, :cyan, :chartreuse, :orchid1]
function plot_wrt_labels(X, y, classes; kwargs...)
    p = plot()
    for (i, c) in enumerate(classes)
        x = X[:, y .== c]
        p = scatter2!(x, color=colorvec[i], label=c; kwargs...)
    end
    return p
end
function plot_wrt_labels!(p, X, y, classes; kwargs...)
    for (i, c) in enumerate(classes)
        x = X[:, y .== c]
        p = scatter2!(x, color=colorvec[i], label=c; kwargs...)
    end
    return p
end

function load_model(par, modelname, r, full, seed)
    d = filter(:r => ri -> ri == r, filter(:full => fi -> fi == full, par[modelname]))
    nm = savename(d.parameters[1])
    modelpath = datadir("experiments", "MNIST", modelname, "seed=$seed")
    files = readdir(modelpath)
    ixs = findall(f -> (occursin(nm, f) && occursin("r=$r", f) && occursin("full=$full", f)), files)
    models = map(x -> BSON.load(joinpath(modelpath, files[x]))[:model], ixs)
    accs = map(x -> BSON.load(joinpath(modelpath, files[x]))[:val_acc], ixs)
    seeds = map(x -> BSON.load(joinpath(modelpath, files[x]))[:seed], ixs)
    ix = findmax(accs)[2]
    model = models[ix]
    return model, ix, d.parameters
end
function load_models(par, modelname, r, full, max_seed)
    d = filter(:r => ri -> ri == r, filter(:full => fi -> fi == full, par[modelname]))
    nm = savename(d.parameters[1])
    modelpathf(seed) = datadir("experiments", "MNIST", modelname, "seed=$seed")
    files = mapreduce(seed -> readdir(modelpathf(seed)), vcat, 1:max_seed)
    ixs = findall(f -> (occursin(nm, f) && occursin("r=$r", f) && occursin("full=$full", f)), files)

    files = sort!(files[ixs])
    loaded_files = map((f, i) -> BSON.load(joinpath(modelpathf(i), f)), files, 1:max_seed)
    models = map(i -> loaded_files[i][:model], 1:max_seed)
    val_accs = map(i -> loaded_files[i][:val_acc], 1:max_seed)
    test_accs = map(i -> loaded_files[i][:test_acc], 1:max_seed)
    seeds = map(i -> loaded_files[i][:seed], 1:max_seed)
    
    return models, seeds, val_accs, test_accs, d.parameters
end
function load_models_arcface(par, modelname, r, full, max_seed)
    d = filter(:r => ri -> ri == r, filter(:full => fi -> fi == full, par[modelname]))
    nm = savename(d.parameters[1])
    modelpathf(seed) = datadir("experiments", "MNIST", modelname, "seed=$seed")
    files = mapreduce(seed -> readdir(modelpathf(seed)), vcat, 1:max_seed)
    ixs = findall(f -> (occursin(nm, f) && occursin("r=$r", f)), files)

    files = sort!(files[ixs])
    loaded_files = map((f, i) -> BSON.load(joinpath(modelpathf(i), f)), files, 1:max_seed)
    models = map(i -> loaded_files[i][:model], 1:max_seed)
    val_accs = map(i -> loaded_files[i][:val_acc], 1:max_seed)
    test_accs = map(i -> loaded_files[i][:test_acc], 1:max_seed)
    seeds = map(i -> loaded_files[i][:seed], 1:max_seed)
    
    return models, seeds, val_accs, test_accs, d.parameters
end


using master_thesis: dist_knn

function try_catch_knn(k, DM, yk, y)
    try
        knn_v, kv = findmax(k -> dist_knn(k, DM, yk, y)[2], 1:k)
        return knn_v, kv
    catch e
        @warn "Caught error, reduced k: $k -> $(k-1)"
        try_catch_knn(k-1, DM, yk, y)
    end
end

"""
    knn(k::Int, enc, enc_val, enc_test, y, yval, yt; kmax=15)

kNN on given embeddings. Best `k` is chosen from `1:kmax` based on accuracy
on validation data.
"""
function knn(enc, enc_val, enc_test, y, yval, yt; kmax=15, type="")
    dm_val = pairwise(Euclidean(), enc_val, enc)
    # @info "Finding best k."
    # val_acc, kbest = findmax(k -> dist_knn(k, dm_val, y, yval)[2], 1:kmax)
    val_acc, kbest = try_catch_knn(kmax, dm_val, y, yval)
    # @info "Best k is $kbest."

    dm_test = pairwise(Euclidean(), enc_test, enc)
    # @info size(dm_test)
    ynew, test_acc = dist_knn(kbest, dm_test, y, yt)

    # return kbest, val_acc, test_acc
    DataFrame(
        :method => "kNN",
        :k_neighbors => kbest,
        :accuracy => test_acc,
        :new_randindex => randindex(yt, ynew)[2],
        :new_adj_randindex => randindex(yt, ynew)[1],
        :new_MI => mutualinfo(yt, ynew),
        :type => type
    )
end

# dm_val = pairwise(Euclidean(), enc_val, enc)
# dm_test = pairwise(Euclidean(), enc_test, enc)
function knn(dm_val, dm_test, y, yval, yt; kmax=15, type="")
    # val_acc, kbest = findmax(k -> dist_knn(k, dm_val, y, yval)[2], 1:kmax)  
    val_acc, kbest = try_catch_knn(kmax, dm_val, y, yval)
    ynew, test_acc = dist_knn(kbest, dm_test, y, yt)
    
    # return kbest, val_acc, test_acc
    DataFrame(
        :method => "kNN",
        :k_neighbors => kbest,
        :accuracy => test_acc,
        :new_randindex => randindex(yt, ynew)[2],
        :new_adj_randindex => randindex(yt, ynew)[1],
        :new_MI => mutualinfo(yt, ynew),
        :type => type
    )
end