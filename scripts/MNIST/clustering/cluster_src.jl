"""
1. Load the data based on the parameters given: r, full.
2. Load the best model.
3. Create a function `embedding(X)`.
4. Use prepared functions to calculate clusterings, umaps etc.
"""

using Clustering, Distances
using ProgressMeter

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
accuracy(y1, y2) = mean(y1 .== y2)

function cluster(DM::AbstractMatrix, y::AbstractVector, clusterfun, k; iter=5)
    max_silh = 0
    cbest = ClusteringResult

    if clusterfun in [kmedoids, kmeans]
        @showprogress "Clustering with $k clusters\n:" for i in 1:iter
            c = clusterfun(DM, k)
            m = mean(silhouettes(c, DM))
            if m > max_silh
                cbest = c
                max_silh = m
            end
        end
    else
        cbest = clusterfun(DM, k)
        max_silh = mean(silhouettes(cbest, DM))
    end
    
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
function cluster(enc, enc_test, y, yt, clusterfun, k, seed; advanced=true, type="", iter=5)
    advanced ? mapfun = map_clustering_advanced : mapfun = map_clustering

    DM = pairwise(Euclidean(), enc_test)
    df, c = cluster(DM, yt, clusterfun, k; iter=iter)
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
function cluster(enc, enc_test, DM, y, yt, clusterfun, k, seed; advanced=true, type="", iter=5)
    advanced ? mapfun = map_clustering_advanced : mapfun = map_clustering
    
    df, c = cluster(DM, yt, clusterfun, k; iter=iter)
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
function plot_wrt_labels(X, y, classes; kwargs...)
    p = plot()
    for (i, c) in enumerate(classes)
        x = X[:, y .== c]
        p = scatter2!(x, color=i, label=c; kwargs...)
    end
    return p
end
function plot_wrt_labels!(p, X, y, classes; kwargs...)
    for (i, c) in enumerate(classes)
        x = X[:, y .== c]
        p = scatter2!(x, color=i, label=c; kwargs...)
    end
    return p
end

function load_model(par, modelname, r, full)
    d = filter(:r => ri -> ri == r, filter(:full => fi -> fi == full, par[modelname]))
    nm = savename(d.parameters[1])
    modelpath = datadir("experiments", "MNIST", modelname)
    files = readdir(modelpath)
    ixs = findall(f -> (occursin(nm, f) && occursin("r=$r", f) && occursin("full=$full", f)), files)
    models = map(x -> BSON.load(joinpath(modelpath, files[x]))[:model], ixs)
    accs = map(x -> BSON.load(joinpath(modelpath, files[x]))[:val_acc], ixs)
    seeds = map(x -> BSON.load(joinpath(modelpath, files[x]))[:seed], ixs)
    ix = findmax(accs)[2]
    model = models[ix]
    return model, ix, d.parameters
end
function load_models(par, modelname, r, full)
    d = filter(:r => ri -> ri == r, filter(:full => fi -> fi == full, par[modelname]))
    nm = savename(d.parameters[1])
    modelpath = datadir("experiments", "MNIST", modelname)
    files = readdir(modelpath)
    ixs = findall(f -> (occursin(nm, f) && occursin("r=$r", f) && occursin("full=$full", f)), files)

    models = map(x -> BSON.load(joinpath(modelpath, files[x]))[:model], ixs)
    val_accs = map(x -> BSON.load(joinpath(modelpath, files[x]))[:val_acc], ixs)
    test_accs = map(x -> BSON.load(joinpath(modelpath, files[x]))[:test_acc], ixs)
    seeds = map(x -> BSON.load(joinpath(modelpath, files[x]))[:seed], ixs)
    
    return models, seeds, val_accs, test_accs, d.parameters
end


using master_thesis: dist_knn

"""
    knn(k::Int, enc, enc_val, enc_test, y, yval, yt; kmax=15)

kNN on given embeddings. Best `k` is chosen from `1:kmax` based on accuracy
on validation data.
"""
function knn(enc, enc_val, enc_test, y, yval, yt; kmax=15, type="")
    dm_val = pairwise(Euclidean(), enc_val, enc)
    # @info "Finding best k."
    val_acc, kbest = findmax(k -> dist_knn(k, dm_val, y, yval)[2], 1:kmax)
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
    val_acc, kbest = findmax(k -> dist_knn(k, dm_val, y, yval)[2], 1:kmax)  
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