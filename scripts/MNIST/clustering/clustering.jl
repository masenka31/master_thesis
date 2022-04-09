using DrWatson
@quickactivate

include(scriptsdir("MNIST", "run_scripts", "results.jl"))

using BSON
# find the model with best validation accuracy over seeds
par = Dict("classifier" => df_c, "classifier_triplet" => df_tr, "M2" => df_m)

modelname = "classifier_triplet"
r, full = 0.002, true
d = filter(:r => ri -> ri == r, filter(:full => fi -> fi == full, par[modelname]))
nm = savename(d.parameters[1])
modelpath = datadir("experiments", "MNIST", modelname)
files = readdir(modelpath)
ixs = findall(f -> (occursin(nm, f) && occursin("r=$r", f) && occursin("full=$full", f)), files)
models = map(x -> BSON.load(joinpath(modelpath, files[x]))[:model], ixs)
accs = map(x -> BSON.load(joinpath(modelpath, files[x]))[:val_acc], ixs)
ix = findmax(accs)[2]
model = models[ix]

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
    return model, ix
end

using Plots
ENV["GKSwstype"] = "100"
gr(;markerstrokewidth=0, label="", color=:jet)

using master_thesis, StatsBase
using master_thesis: seqids2bags, reindex, encode

using UMAP
using UMAP: transform

include(srcdir("point_cloud.jl"))
data = load_mnist_point_cloud()

ratios = (r, 0.5-r, 0.5)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.bag_labels; ratios=ratios, seed=(1:5)[ix])

if !full
    b = map(x -> any(x .== [0,1,3,4]), data.bag_labels)
    filt_data, filt_labels = reindex(data.data, b), data.bag_labels[b]
    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(filt_data, filt_labels; ratios=ratios, seed=(1:5)[ix])
end

classes = sort(unique(yk))
c = n = length(classes)

if modelname in ["classifier", "classifier_triplet"]
    embedding(x) = Mill.data(model[1](x))
end
enc = embedding(Xk)
enct = embedding(Xt)

nn = 15

# UMAP model on train data
umap_model = UMAP_(enc, 2, n_neighbors=nn)
emb = umap_model.embedding
emb2 = transform(umap_model, enct)

p = plot_wrt_labels(emb2, yt, classes; ms=2, legend=:outerright)
p = plot_wrt_labels!(p, emb, yk, classes; marker=:square, markerstrokewidth=1)

# UMAP model on test data
umap_test = UMAP_(enct, 2, n_neighbors=nn)
emb_model_test = umap_test.embedding
emb_model_train = transform(umap_test, enc)

p = plot_wrt_labels(emb_model_test, yt, classes; ms=2, legend=:outerright)
p = plot_wrt_labels!(p, emb_model_train, yk, classes; marker=:square, markerstrokewidth=1)

# simple HMill encoding
ye = encode(yk, classes)
yet = encode(yt, classes)

len = length(classes)
ks = len .* [1,2,3]

include(scriptsdir("MNIST", "clustering", "cluster_src.jl"))

result_means = map(k -> cluster(enc, enct, ye, yet, kmeans, k), ks)
result_medoids = map(k -> cluster(enc, enct, ye, yet, kmedoids, k), ks)
result_hs = map(k -> cluster(enc, enct, ye, yet, hierarchical_single, k), ks)
result_ha = map(k -> cluster(enc, enct, ye, yet, hierarchical_average, k), ks)

result_means_umap_train = map(k -> cluster(emb, emb2, ye, yet, kmeans, k), ks)
result_medoids_umap_train = map(k -> cluster(emb, emb2, ye, yet, kmedoids, k), ks)
result_hs_umap_train = map(k -> cluster(emb, emb2, ye, yet, hierarchical_single, k), ks)
result_ha_umap_train = map(k -> cluster(emb, emb2, ye, yet, hierarchical_average, k), ks)

result_means_umap_test = map(k -> cluster(emb_model_train, emb_model_test, ye, yet, kmeans, k), ks)
result_medoids_umap_test = map(k -> cluster(emb_model_train, emb_model_test, ye, yet, kmedoids, k), ks)
result_hs_umap_test = map(k -> cluster(emb_model_train, emb_model_test, ye, yet, hierarchical_single, k), ks)
result_ha_umap_test = map(k -> cluster(emb_model_train, emb_model_test, ye, yet, hierarchical_average, k), ks)

res_vec = [
    [result_means, result_medoids, result_hs, result_ha],
    [result_means_umap_train, result_medoids_umap_train, result_hs_umap_train, result_ha_umap_train],
    [result_means_umap_test, result_medoids_umap_test, result_hs_umap_test, result_ha_umap_test]
]

vcat(map(res -> mapreduce(i -> res[i][1], vcat, 1:3), res_vec[1])..., cols=:union)
vcat(map(res -> mapreduce(i -> res[i][1], vcat, 1:3), res_vec[2])..., cols=:union)
vcat(map(res -> mapreduce(i -> res[i][1], vcat, 1:3), res_vec[3])..., cols=:union)