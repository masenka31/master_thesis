using DrWatson
@quickactivate

# include source files (this file actually calls another file in MNIST scripts folder)
include(scriptsdir("MIProblems", "clustering", "cluster_src.jl"))
include(srcdir("point_cloud.jl"))
include(srcdir("mill_data.jl"))

function embeddingfun(modelname, model)
    if modelname in ["classifier", "classifier_triplet", "self_classifier"]
        return x -> Mill.data(model[1](x))
    elseif modelname == "M2"
        return x -> model.bagmodel(x)
    elseif modelname == "self_arcface"
        return x -> model(x)
    end
end

# load data tables
include(scriptsdir("MIProblems", "run_scripts", "results_animals.jl"))
par = Dict("classifier" => classifier, "classifier_triplet" => triplet, "M2" => m2, "self_classifier" => self, "self_arcface" => arc)

# include the clustering functions
include(scriptsdir("MIProblems", "clustering", "majority_voting.jl"))

modelname = ARGS[1]
r = parse(Float64, ARGS[2])
dataset = "animals_negative"
# r, modelname, dataset = 0.05, "classifier_triplet", "animals_negative"
ratios = (r, 0.5-r, 0.5)
models, seeds, val_accs, test_accs, params = load_models(par, dataset, modelname, r)

function calculate_majority(models, seeds, val_accs, test_accs, params, dataset, ratios, modelname, r)
    results_over_seeds = DataFrame[]

    for (i, seed) in enumerate(seeds)
        @info "Calculating seed $seed."
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

        enc_full = hcat(enc, enct)

        @info "Embeddings calculated."

        # choose number of nearest neighbors
        nn = 10
        r == 0.05 ? nn = 5 : nothing
        @info "Neighbors: $nn."

        # UMAP model on train data
        emb = umap(enc_full, 2, n_neighbors=nn)
        
        # ixx = vcat(zeros(Int, length(yk)), ones(Int, length(yt))) .+ 1
        # ms = map(i -> [:square, :circle][i], ixx)
        # if seed < 3
        #     p = plot_wrt_labels(emb, vcat(yk, yt), classes; ms=4, legend=:outerright)
        #     # p = plot_wrt_labels!(p, emb_val1, yval, classes; marker=:diamond, markerstrokewidth=1)
        #     mkpath(plotsdir("MIProblems2", modelname, "seed=$seed"))
        #     savefig(plotsdir("MIProblems2", modelname, "seed=$seed", "train_umap_r=$(r)_val_acc=$(val_accs[i])_test_acc=$(test_accs[i]).png"))
        # end

        @info "UMAP embeddings calculated and plots saved."

        ################################################################
        ###                        Clustering                        ###
        ################################################################

        # initialize number of clusters
        len = length(classes)
        ks = len .* [1,2,3]

        # use parallel computing on number of clusters ks
        # load the same number of threads as length of ks

        results_kmeans = mapreduce(k -> try_catch_cluster(enc_full, kmeans, yk, yt, k), vcat, ks)
        results_kmedoids = mapreduce(k -> try_catch_cluster(enc_full, kmedoids, yk, yt, k), vcat, ks)
        results_hier = mapreduce(k -> try_catch_cluster(enc_full, hierarchical_average, yk, yt, k), vcat, ks)

        results_kmeans_emb = mapreduce(k -> try_catch_cluster(emb, kmeans, yk, yt, k), vcat, ks)
        results_kmedoids_emb = mapreduce(k -> try_catch_cluster(emb, kmedoids, yk, yt, k), vcat, ks)
        results_hier_emb = mapreduce(k -> try_catch_cluster(emb, hierarchical_average, yk, yt, k), vcat, ks)
            
        @info "Clusterings calculated."

        # results
        results = vcat(results_kmeans, results_kmedoids, results_hier, results_kmeans_emb, results_kmedoids_emb, results_hier_emb)
        typenames = vcat(repeat(["encoding"], 9), repeat(["umap"], 9))
        results.type = typenames

        ###########
        ### kNN ###
        ###########
        
        yk, yval, yt = encode(yk, classes), encode(yval, classes), encode(yt, classes)
        encoding_knn = knn(enc, encv, enct, yk, yval, yt; type="encoding", kmax=nn)
        eknn = encoding_knn[:, [:method, :accuracy, :type, :k_neighbors]]
        rename!(eknn, :k_neighbors => :k)
        full_results = vcat(results, eknn, cols=:union)

        # Put all results together and push to results dataframe vector
        push!(results_over_seeds, full_results)
    end
    return results_over_seeds
end

R = calculate_majority(models, seeds, val_accs, test_accs, params, dataset, ratios, modelname, r)

rdf = vcat(R...)
rdf = hcat(DataFrame(:modelname => repeat([modelname], nrow(rdf))), rdf)
gdf = groupby(rdf, [:modelname, :method, :k, :type])
cdf = combine(
    gdf,
    [:randindex, :adj_randindex, :MI, :silh, :accuracy] .=> mean,
    renamecols=false
)
pretty_table(
    cdf, nosubheader=true,
    hlines=vcat(0,1,4:3:40...),
    formatters = ft_round(3), crop=:none
)

results_dict = Dict(
    :full_df => rdf,
    :combined_df => cdf,
    :parameters => params,
    :r => r,
    :models => models,
    :modelname => modelname
)

safesave(datadir("experiments", "MIProblems2", "clustering", modelname, savename(results_dict, "bson")), results_dict)
