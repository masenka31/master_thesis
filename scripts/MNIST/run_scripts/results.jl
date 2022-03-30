using DrWatson
@quickactivate

using DataFrames, Flux, Mill, PrettyTables
using Statistics

df = collect_results(datadir("experiments", "MNIST"), subfolders=true)

g_model = groupby(df, :modelname)

function get_results_table(df)
    # group to different experiments
    # full - how many digits, r - how many labels, parameters - controls the model
    # average results over seeds
    g = groupby(df, [:full, :r, :parameters])
    @info "Got $(length(g)) groups."

    # filter out experiments that did not run over full 5 seeds
    f = filter(g -> nrow(g) == 5, g)
    @info "$(length(f)) groups left after filtering."
    c = combine(f, [:train_acc, :val_acc, :test_acc] .=> mean, renamecols=false)
    par_df = DataFrame(c.parameters)
    # d = sort(hcat(c[:, Not(:parameters)], par_df), :val_acc, rev=true)
    d = sort(hcat(c, par_df), :val_acc, rev=true)

    gg = groupby(d, [:full, :r])

    # just a check that the right rows are selected
    if sum(map(i -> findmax(gg[i].val_acc)[2], 1:6) .!= 1) != 0
        error("Check failed, wrong rows would be selected.")
    else
        @info "Check passed. Right rows selected."
    end

    best = mapreduce(i -> DataFrame(gg[i][1, :]), vcat, 1:length(gg))
    sort(best, [:full, :r])
end

df_c = get_results_table(g_model[1])
df_tr = get_results_table(g_model[2])

f1 = filter(:r => x -> x == 0.002, df)
f1 = filter(:full => x -> x == true, f1)
i = 4 # 1 for 4 classes, 4 for full
fc = sort(filter(:parameters => x -> x == df_c[i, :parameters], f1), :val_acc, rev=true)
ftr = sort(filter(:parameters => x -> x == df_tr[i, :parameters], f1), :val_acc, rev=true)

model_c = fc[1, :model]
model_tr = ftr[1, :model]

using Plots
ENV["GKSwstype"] = "100"
gr(;markerstrokewidth=0, label="", color=:jet)

using master_thesis, StatsBase
using master_thesis: seqids2bags, reindex

include(srcdir("point_cloud.jl"))
data = load_mnist_point_cloud()

r = 0.002
ratios = (r, 0.5-r, 0.5)
seed = 5
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.bag_labels; ratios=ratios, seed=seed)

b = map(x -> any(x .== [0,1,3,4]), data.bag_labels)
filt_data, filt_labels = reindex(data.data, b), data.bag_labels[b]
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(filt_data, filt_labels; ratios=ratios, seed=seed)

X, y = Xt, yt
enc = model_c[1:end-1](X)
enc = model_tr[1:end-1](X)
p = scatter2(enc, zcolor=y, ms=2)
wsave(plotsdir("encoding_c.svg"), p)

using UMAP
@time emb = umap(enc, 2, n_neighbors=10)
p = scatter2(emb, zcolor=y, ms=2)
wsave(plotsdir("embedding_c.svg"), p)


using Clustering, Distances

function cluster_data(X, y, model, c)
    enc = model[1:end-1](X)
    c1 = kmedoids(pairwise(Euclidean(), enc), c)
    ri1 = randindex(assignments(c1), y)[1]

    c2 = cutree(hclust(pairwise(Euclidean(), enc), linkage=:average), k=c)
    ri2 = randindex(c2, y)[1]

    p = scatter2(enc, zcolor=assignments(c1), color=:jet)
    wsave(plotsdir("cluster.svg"), p)
    
    p = scatter2(enc, zcolor=c2, color=:jet)
    wsave(plotsdir("cluster2.svg"), p)

    return (kmeans = ri1, hclust = ri2)
end

c = 10
cluster_data(Xk,yk,model_c,c)
cluster_data(Xt,yt,model_c,c)

cluster_data(Xk,yk,model_tr,c)
cluster_data(Xt,yt,model_tr,c)