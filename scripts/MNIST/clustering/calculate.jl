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
model, ix, params = load_model(par, modelname, r, fl)
seed = (1:5)[ix]

# ininialize data
data = load_mnist_point_cloud()
ratios = (r, 0.5-r, 0.5)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.bag_labels; ratios=ratios, seed=seed)

if !fl
    b = map(x -> any(x .== [0,1,3,4]), data.bag_labels)
    filt_data, filt_labels = reindex(data.data, b), data.bag_labels[b]
    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(filt_data, filt_labels; ratios=ratios, seed=seed)
end

classes = sort(unique(yk))
c = n = length(classes)

if modelname in ["classifier", "classifier_triplet"]
    embedding(x) = Mill.data(model[1](x))
elseif modelname == "M2"
    embedding(x) = Mill.data(model.bagmodel(x))
end

# calculate the encodings of HMill model
enc = embedding(Xk)
enct = embedding(Xt)
# label encoding
ye = encode(yk, classes)
yet = encode(yt, classes)

# choose number of nearest neighbors
nn = 15

# UMAP model on train data
tr_model = UMAP_(enc, 2, n_neighbors=nn)
emb = tr_model.embedding
emb_test = UMAP.transform(tr_model, enct)

p = plot_wrt_labels(emb_test, yt, classes; ms=2, legend=:outerright)
p = plot_wrt_labels!(p, emb, yk, classes; marker=:square, markerstrokewidth=1)
savefig(plotsdir("MNIST", modelname, "train_umap_r=$(r)_full=$(fl)_seed=$seed.png"))

# UMAP model on test data
ts_model = UMAP_(enct, 2, n_neighbors=nn)
emb2 = ts_model.embedding
emb_train = UMAP.transform(ts_model, enc)

p = plot_wrt_labels(emb2, yt, classes; ms=2, legend=:outerright)
p = plot_wrt_labels!(p, emb_train, yk, classes; marker=:square, markerstrokewidth=1)
savefig(plotsdir("MNIST", modelname, "test_umap_r=$(r)_full=$(fl)_seed=$seed.png"))

################################################################
###                        Clustering                        ###
################################################################

# initialize number of clusters
len = length(classes)
ks = len .* [1,2,3]
       
# results on simple encoding
result_means = map(k -> cluster(enc, enct, ye, yet, kmeans, k), ks)
result_medoids = map(k -> cluster(enc, enct, ye, yet, kmedoids, k), ks)
result_hs = map(k -> cluster(enc, enct, ye, yet, hierarchical_single, k), ks)
result_ha = map(k -> cluster(enc, enct, ye, yet, hierarchical_average, k), ks)

# results on UMAP calculated on train data
result_means_umap_train = map(k -> cluster(emb, emb_test, ye, yet, kmeans, k), ks)
result_medoids_umap_train = map(k -> cluster(emb, emb_test, ye, yet, kmedoids, k), ks)
result_hs_umap_train = map(k -> cluster(emb, emb_test, ye, yet, hierarchical_single, k), ks)
result_ha_umap_train = map(k -> cluster(emb, emb_test, ye, yet, hierarchical_average, k), ks)

# results on UMAP calculated on test data
result_means_umap_test = map(k -> cluster(emb_train, emb2, ye, yet, kmeans, k), ks)
result_medoids_umap_test = map(k -> cluster(emb_train, emb2, ye, yet, kmedoids, k), ks)
result_hs_umap_test = map(k -> cluster(emb_train, emb2, ye, yet, hierarchical_single, k), ks)
result_ha_umap_test = map(k -> cluster(emb_train, emb2, ye, yet, hierarchical_average, k), ks)

# all results together
res_vec = [
    [result_means, result_medoids, result_hs, result_ha],
    [result_means_umap_train, result_medoids_umap_train, result_hs_umap_train, result_ha_umap_train],
    [result_means_umap_test, result_medoids_umap_test, result_hs_umap_test, result_ha_umap_test]
]

encoding_results = vcat(map(res -> mapreduce(i -> res[i][1], vcat, 1:3), res_vec[1])..., cols=:union)
train_umap_results = vcat(map(res -> mapreduce(i -> res[i][1], vcat, 1:3), res_vec[2])..., cols=:union)
test_umap_results = vcat(map(res -> mapreduce(i -> res[i][1], vcat, 1:3), res_vec[3])..., cols=:union)

par_name = savename(parameters)
savepath = datadir("experiments", "MNIST", "clustering", modelname)

safesave(savename(par_name, "encoding_results_seed=$seed", "bson"), Dict(:results => encoding_results, :parameters => params, :model => model))
safesave(savename(par_name, "train_umap_results_seed=$seed", "bson"), Dict(:results => train_umap_results))
safesave(savename(par_name, "test_umap_results_seed=$seed", "bson"), Dict(:results => test_umap_results))
