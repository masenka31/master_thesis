using UMAP
using UMAP: UMAP.transform
using Distributed

using Plots
ENV["GKSwstype"] = "100"
gr(;markerstrokewidth=0, label="", color=:jet)

using BSON, StatsBase
using ThreadTools

function load_models(par::Dict, modelname::String, r::Number)
    d = filter(:r => ri -> ri == r, par[modelname])
    nm = savename(d.parameters[1])
    modelpath = datadir("experiments", "gvma", modelname)
    modelpathf(seed) = datadir("experiments", "gvma", "seed=$seed")

    F = []
    for seed in 1:6
        p = datadir("experiments", "gvma",modelname, "seed=$seed")
        files = readdir(p)
        files_filt = filter(f -> (occursin("r=$(r)_", f) && occursin(nm, f)), files)
        files = map(f -> joinpath(p, f), files_filt)
        push!(F, files)
    end
    files = vcat(F...)
    loaded_files = map((f, i) -> BSON.load(joinpath(modelpathf(i), f)), files, 1:6)

    models = map(f -> f[:model], loaded_files);
    val_accs = map(f -> f[:val_acc], loaded_files)
    test_accs = map(f -> f[:test_acc], loaded_files)
    seeds = map(f -> f[:seed], loaded_files)
    
    return models, seeds, val_accs, test_accs, d.parameters
end

"""
1. Load the data based on the parameters given: r, full.
2. Load the best model.
3. Create a function `embedding(X)`.
4. Use prepared functions to calculate clusterings, umaps etc.
"""

using Clustering, Distances

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

using gvma: dist_knn
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
function knn(dm_val, dm_test, y, yval, yt; kmax=15, type="")
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

using gvma: encode
function calculate_clustering_results(models, seeds, val_accs, test_accs, params, X, y, ratios, modelname, r)
    results_over_seeds = DataFrame[]

    for (i, seed) in enumerate(seeds)
        # load data
        Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(X, y; ratios=ratios, seed=seed)
        classes = sort(unique(yk))
        n = c = length(classes)
        Xval, yval, Xu, yu = validation_data(yk, Xu, yu, seed, classes)
        @info "Data prepared."

        model = models[i];
        @info modelname
        embedding = embeddingfun(modelname, model)

        # calculate the encodings of HMill model
        enc = embedding(Xk)
        encv = embedding(Xval)
        enct = embedding(Xt)
        # label encoding
        ye = encode(yk, classes)
        yev = encode(yval, classes)
        yet = encode(yt, classes)

        @info "Embeddings calculated."

        # choose number of nearest neighbors
        nn = 15
        @info "Neighbors: $nn."

        # UMAP model on train data
        tr_model = UMAP_(enc, 2; n_neighbors=nn)
        emb = tr_model.embedding
        emb_test = UMAP.transform(tr_model, enct; n_neighbors=nn)
        emb_val1 = UMAP.transform(tr_model, encv; n_neighbors=nn)

        # UMAP model on test data
        ts_model = UMAP_(enct, 2; n_neighbors=nn)
        emb2 = ts_model.embedding
        emb_train = UMAP.transform(ts_model, enc; n_neighbors=nn)
        emb_val2 = UMAP.transform(ts_model, encv; n_neighbors=nn)

        if seed < 6
            p = plot_wrt_labels(emb_test, yt, classes; ms=4, legend=:outerright)
            p = plot_wrt_labels!(p, emb, yk, classes; m=:square, ms=3, markerstrokewidth=1)
            # p = plot_wrt_labels!(p, emb_val1, yval, classes; marker=:diamond, markerstrokewidth=1)
            mkpath(plotsdir("gvma", modelname, "seed=$seed"))
            savefig(plotsdir("gvma", modelname, "seed=$seed", "train_umap_r=$(r)_val_acc=$(val_accs[i])_test_acc=$(test_accs[i]).png"))

            p = plot_wrt_labels(emb2, yt, classes; ms=4, legend=:outerright)
            p = plot_wrt_labels!(p, emb_train, yk, classes; ms=3, marker=:square, markerstrokewidth=1)
            savefig(plotsdir("gvma", modelname, "seed=$seed", "test_umap_r=$(r)_val_acc=$(val_accs[i])_test_acc=$(test_accs[i]).png"))
        end

        @info "UMAP embeddings calculated and plots saved."

        ################################################################
        ###                        Clustering                        ###
        ################################################################

        # initialize number of clusters
        len = length(classes)
        ks = len .* [1,2,3]

        # use parallel computing on number of clusters ks
        # load the same number of threads as length of ks
            
        # results on simple encoding
        DM1 = pairwise(Euclidean(), enct)
        result_means = tmap(k -> cluster(enc, enct, DM1, ye, yet, kmeans, k, seed; type = "encoding"), 3, ks)
        result_medoids = tmap(k -> cluster(enc, enct, DM1, ye, yet, kmedoids, k, seed; type = "encoding"), 3, ks)
        result_ha = tmap(k -> cluster(enc, enct, DM1, ye, yet, hierarchical_average, k, seed; type = "encoding"), 3, ks)

        @info "Clustering results on HMill encoding calculated."

        # results on UMAP calculated on train data
        DM2 = pairwise(Euclidean(), emb_test)
        result_means_umap_train = tmap(k -> cluster(emb, emb_test, DM2, ye, yet, kmeans, k, seed; type = "train_embedding"), 3, ks)
        result_medoids_umap_train = tmap(k -> cluster(emb, emb_test, DM2, ye, yet, kmedoids, k, seed; type = "train_embedding"), 3, ks)
        result_ha_umap_train = tmap(k -> cluster(emb, emb_test, DM2, ye, yet, hierarchical_average, k, seed; type = "train_embedding"), 3, ks)

        @info "Clustering results on train embedding calculated."

        # results on UMAP calculated on test data
        DM3 = pairwise(Euclidean(), emb2)
        result_means_umap_test = tmap(k -> cluster(emb_train, emb2, DM3, ye, yet, kmeans, k, seed; type = "test_embedding"), 3, ks)
        result_medoids_umap_test = tmap(k -> cluster(emb_train, emb2, DM3, ye, yet, kmedoids, k, seed; type = "test_embedding"), 3, ks)
        result_ha_umap_test = tmap(k -> cluster(emb_train, emb2, DM3, ye, yet, hierarchical_average, k, seed; type = "test_embedding"), 3, ks)

        @info "Clustering results on test embedding calculated."

        # all results together
        res_vec = [
            [result_means, result_medoids, result_ha],
            [result_means_umap_train, result_medoids_umap_train, result_ha_umap_train],
            [result_means_umap_test, result_medoids_umap_test, result_ha_umap_test]
        ];

        encoding_results = vcat(map(res -> mapreduce(k -> res[k][1], vcat, 1:3), res_vec[1])..., cols=:union)
        train_umap_results = vcat(map(res -> mapreduce(k -> res[k][1], vcat, 1:3), res_vec[2])..., cols=:union)
        test_umap_results = vcat(map(res -> mapreduce(k -> res[k][1], vcat, 1:3), res_vec[3])..., cols=:union)
        _full_results = vcat(encoding_results, train_umap_results, test_umap_results, cols=:union)
        nd = nrow(_full_results)
        full_results = hcat(_full_results, DataFrame(:val_acc => repeat([val_accs[i]], nd), :test_acc => repeat([test_accs[i]], nd)))

        ###########
        ### kNN ###
        ###########

        yk, yval, yt = encode(yk, classes), encode(yval, classes), encode(yt, classes)
        encoding_knn = knn(enc, encv, enct, yk, yval, yt; type="encoding", kmax=nn)
        train_umap_knn = knn(emb, emb_val1, emb_test, yk, yval, yt, type="train_embedding")
        test_umap_knn = knn(emb_train, emb_val2, emb2, yk, yval, yt, type="test_embedding")
        knn_results = vcat(encoding_knn, train_umap_knn, test_umap_knn)

        # Put all results together and push to results dataframe vector
        R = vcat(full_results, knn_results, cols=:union)
        push!(results_over_seeds, R)
    end
    return results_over_seeds
end