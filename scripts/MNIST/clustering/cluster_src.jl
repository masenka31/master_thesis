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
function map_clustering(enc_train, enc_test, ytrain, assign; metric=:mse)
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
function cluster(enc, enc_test, y, yt, clusterfun, k; iter=5)
    DM = pairwise(Euclidean(), enc_test)
    df, c = cluster(DM, yt, clusterfun, k; iter=iter)
    if clusterfun in [kmedoids, kmeans]
        ynew = map_clustering(enc, enc_test, y, assignments(c))
    else
        ynew = map_clustering(enc, enc_test, y, c)
    end
    a = accuracy(yt, ynew)
    da = DataFrame(
        :accuracy => a,
        :new_randindex => randindex(yt, ynew)[2],
        :new_ajd_randindex => randindex(yt, ynew)[1],
        :new_MI => mutualinfo(yt, ynew)
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
    ix = findmax(accs)[2]
    model = models[ix]
    return model, ix, d.parameters
end