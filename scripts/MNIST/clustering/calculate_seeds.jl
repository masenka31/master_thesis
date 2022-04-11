using DrWatson
@quickactivate

using BSON
using Plots
ENV["GKSwstype"] = "100"
gr(;markerstrokewidth=0, label="", color=:jet)

using master_thesis, StatsBase
using master_thesis: seqids2bags, reindex, encode

using UMAP
using UMAP: UMAP.transform
using Distributed
using master_thesis: dist_knn

include(scriptsdir("MNIST", "run_scripts", "results.jl"))
include(srcdir("point_cloud.jl"))
include(scriptsdir("MNIST", "clustering", "cluster_src.jl"))

### ARGS
if isempty(ARGS)
    modelname, r, fl = "classifier", 0.002, false
else
    modelname = ARGS[1]
    r = parse(Float64, ARGS[2])
    fl = parse(Bool, ARGS[3])
end

# load the model
par = Dict("classifier" => df_c, "classifier_triplet" => df_tr, "M2" => df_m)
models, seeds, val_accs, test_accs, params = load_models(par, modelname, r, fl)

# ininialize data
data = load_mnist_point_cloud()
ratios = (r, 0.5-r, 0.5)

function embeddingfun(modelname, model)
    if modelname in ["classifier", "classifier_triplet"]
        return x -> Mill.data(model[1](x))
    elseif modelname == "M2"
        return x -> Mill.data(model.bagmodel(x))
    end
end

function calculate_clustering_results(models, seeds, val_accs, test_accs, params, data, ratios, modelname, r, fl)
    results_over_seeds = DataFrame[]

    for (i, seed) in enumerate(seeds)
        if fl
            Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.bag_labels; ratios=ratios, seed=seed)
        else
            b = map(x -> any(x .== [0,1,3,4]), data.bag_labels)
            filt_data, filt_labels = reindex(data.data, b), data.bag_labels[b]
            Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(filt_data, filt_labels; ratios=ratios, seed=seed)
        end

        classes = sort(unique(yk))
        c = n = length(classes)
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
        nn = 15

        # UMAP model on train data
        tr_model = UMAP_(enc, 2, n_neighbors=nn)
        emb = tr_model.embedding
        emb_test = UMAP.transform(tr_model, enct)
        emb_val1 = UMAP.transform(tr_model, encv)

        p = plot_wrt_labels(emb_test, yt, classes; ms=2, legend=:outerright)
        p = plot_wrt_labels!(p, emb, yk, classes; marker=:square, markerstrokewidth=1)
        # p = plot_wrt_labels!(p, emb_val1, yval, classes; marker=:diamond, markerstrokewidth=1)
        mkpath(plotsdir("MNIST", modelname))
        savefig(plotsdir("MNIST", modelname, "train_umap_r=$(r)_full=$(fl)_seed=$(seed)_val_acc=$(val_accs[i])_test_acc=$(test_accs[i]).png"))

        # UMAP model on test data
        ts_model = UMAP_(enct, 2, n_neighbors=nn)
        emb2 = ts_model.embedding
        emb_train = UMAP.transform(ts_model, enc)
        emb_val2 = UMAP.transform(ts_model, encv)

        p = plot_wrt_labels(emb2, yt, classes; ms=2, legend=:outerright)
        p = plot_wrt_labels!(p, emb_train, yk, classes; marker=:square, markerstrokewidth=1)
        savefig(plotsdir("MNIST", modelname, "test_umap_r=$(r)_full=$(fl)_seed=$(seed)_val_acc=$(val_accs[i])_test_acc=$(test_accs[i]).png"))

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
        result_means = pmap(k -> cluster(enc, enct, DM1, ye, yet, kmeans, k, seed; type = "encoding"), ks)
        result_medoids = pmap(k -> cluster(enc, enct, DM1, ye, yet, kmedoids, k, seed; type = "encoding"), ks)
        # result_hs = pmap(k -> cluster(enc, enct, DM1, ye, yet, hierarchical_single, k, seed; type = "encoding"), ks)
        result_ha = pmap(k -> cluster(enc, enct, DM1, ye, yet, hierarchical_average, k, seed; type = "encoding"), ks)

        @info "Clustering results on HMill encoding calculated."

        # results on UMAP calculated on train data
        DM2 = pairwise(Euclidean(), emb_test)
        result_means_umap_train = pmap(k -> cluster(emb, emb_test, DM2, ye, yet, kmeans, k, seed; type = "train_embedding"), ks)
        result_medoids_umap_train = pmap(k -> cluster(emb, emb_test, DM2, ye, yet, kmedoids, k, seed; type = "train_embedding"), ks)
        # result_hs_umap_train = pmap(k -> cluster(emb, emb_test, DM2, ye, yet, hierarchical_single, k, seed; type = "train_embedding"), ks)
        result_ha_umap_train = pmap(k -> cluster(emb, emb_test, DM2, ye, yet, hierarchical_average, k, seed; type = "train_embedding"), ks)

        @info "Clustering results on train embedding calculated."

        # results on UMAP calculated on test data
        DM3 = pairwise(Euclidean(), emb2)
        result_means_umap_test = pmap(k -> cluster(emb_train, emb2, DM3, ye, yet, kmeans, k, seed; type = "test_embedding"), ks)
        result_medoids_umap_test = pmap(k -> cluster(emb_train, emb2, DM3, ye, yet, kmedoids, k, seed; type = "test_embedding"), ks)
        # result_hs_umap_test = pmap(k -> cluster(emb_train, emb2, DM3, ye, yet, hierarchical_single, k, seed; type = "test_embedding"), ks)
        result_ha_umap_test = pmap(k -> cluster(emb_train, emb2, DM3, ye, yet, hierarchical_average, k, seed; type = "test_embedding"), ks)

        result_means_umap_test = pmap(k -> cluster(emb_train, emb2, DM3, ye, yet, kmeans, k, seed; type = "test_embedding"), ks)
        result_medoids_umap_test = pmap(k -> cluster(emb_train, emb2, DM3, ye, yet, kmedoids, k, seed; type = "test_embedding"), ks)
        result_ha_umap_test = pmap(k -> cluster(emb_train, emb2, DM3, ye, yet, hierarchical_average, k, seed; type = "test_embedding"), ks)

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

        encoding_knn = knn(enc, encv, enct, yk, yval, yt; type="encoding")
        train_umap_knn = knn(emb, emb_val1, emb_test, yk, yval, yt, type="train_embedding")
        test_umap_knn = knn(emb_train, emb_val2, emb2, yk, yval, yt, type="test_embedding")
        knn_results = vcat(encoding_knn, train_umap_knn, test_umap_knn)

        # Put all results together and push to results dataframe vector
        R = vcat(full_results, knn_results, cols=:union)
        push!(results_over_seeds, R)
    end
    return results_over_seeds
end

R = calculate_clustering_results(models, seeds, val_accs, test_accs, params, data, ratios, modelname, r, fl)
rdf = vcat(R...)
rdf = hcat(DataFrame(:modelname => repeat([modelname], nrow(rdf))), rdf)
gdf = groupby(rdf, [:modelname, :method, :k, :type])
cdf = combine(
    gdf,
    [:randindex, :adj_randindex, :MI, :silh, :accuracy, :new_randindex, :new_adj_randindex, :new_MI, :val_acc, :test_acc] .=> mean,
    renamecols=false
)
pretty_table(
    cdf, nosubheader=true,
    hlines=vcat(0,1,4:3:40...),
    formatters = ft_round(3), crop=:none
)

results_dict = Dict(
    :full_df => :rdf,
    :combined_df => cdf,
    :parameters => params,
    :r => r,
    :full => fl,
    :models => models,
    :modelname => modelname
)

safesave(datadir("experiments", "MNIST", "clustering", modelname, savename(results_dict, "bson")), results_dict)