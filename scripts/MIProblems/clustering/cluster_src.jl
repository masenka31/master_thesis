include(scriptsdir("MNIST", "clustering", "cluster_src.jl"))

using UMAP
using UMAP: UMAP.transform
using Distributed

using Plots
ENV["GKSwstype"] = "100"
gr(;markerstrokewidth=0, label="", color=:jet)

using master_thesis, StatsBase, BSON
using master_thesis: seqids2bags, reindex, encode
using ThreadTools

function load_model(par::Dict, dataset::String, modelname::String, r::Number)
    d = filter(:r => ri -> ri == r, par[modelname])
    nm = savename(d.parameters[1])
    modelpath = datadir("experiments", "MIProblems", dataset, modelname)

    F = []
    for seed in 1:15
        p = datadir("experiments", "MIProblems", dataset, modelname, "seed=$seed")
        files = readdir(p)
        files_filt = filter(f -> (occursin("r=$r", f) && occursin(nm, f)), files)
        files = map(f -> joinpath(p, f), files_filt)
        push!(F, files)
    end
    files = vcat(F...)
    
    models = map(f -> BSON.load(f)[:model], files)
    accs = map(f -> BSON.load(f)[:val_acc], files)
    test_acc = map(f -> BSON.load(f)[:test_acc], files)
    seeds = map(f -> BSON.load(f)[:seed], files)
    
    ix = findmax(accs)[2]
    model = models[ix]
    return model, ix, d.parameters
end
function load_models(par::Dict, dataset::String, modelname::String, r::Number)
    d = filter(:r => ri -> ri == r, par[modelname])
    nm = savename(d.parameters[1])
    modelpath = datadir("experiments", "MIProblems", dataset, modelname)

    F = []
    for seed in 1:15
        p = datadir("experiments", "MIProblems", dataset, modelname, "seed=$seed")
        files = readdir(p)
        files_filt = filter(f -> (occursin("r=$(r)_", f) && occursin(nm, f)), files)
        files = map(f -> joinpath(p, f), files_filt)
        push!(F, files)
    end
    files = vcat(F...)

    models = map(f -> BSON.load(f)[:model], files)
    val_accs = map(f -> BSON.load(f)[:val_acc], files)
    test_accs = map(f -> BSON.load(f)[:test_acc], files)
    seeds = map(f -> BSON.load(f)[:seed], files)
    
    return models, seeds, val_accs, test_accs, d.parameters
end

function calculate_clustering_results(models, seeds, val_accs, test_accs, params, dataset, ratios, modelname, r)
    results_over_seeds = DataFrame[]

    for (i, seed) in enumerate(seeds)
        # load data
        data, labels = load_multiclass(dataset)
        Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data, labels; ratios=ratios, seed=seed)
        classes = sort(unique(yk))
        n = c = length(classes)
        Xval, yval, Xu, yu = validation_data(yk, Xu, yu, seed, classes)
        @info "Data prepared."

        model = models[i]
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
        nn = 10
        r == 0.05 ? nn = 5 : nothing
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
            mkpath(plotsdir("MIProblems", modelname, "seed=$seed"))
            savefig(plotsdir("MIProblems", modelname, "seed=$seed", "train_umap_r=$(r)_val_acc=$(val_accs[i])_test_acc=$(test_accs[i]).png"))

            p = plot_wrt_labels(emb2, yt, classes; ms=4, legend=:outerright)
            p = plot_wrt_labels!(p, emb_train, yk, classes; ms=3, marker=:square, markerstrokewidth=1)
            savefig(plotsdir("MIProblems", modelname, "seed=$seed", "test_umap_r=$(r)_val_acc=$(val_accs[i])_test_acc=$(test_accs[i]).png"))
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