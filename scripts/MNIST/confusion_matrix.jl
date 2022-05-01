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
using ThreadTools

include(scriptsdir("MNIST", "run_scripts", "results.jl"))
include(srcdir("point_cloud.jl"))
include(scriptsdir("MNIST", "clustering", "cluster_src.jl"))

# load the model
par = Dict(
    "classifier" => df_c,
    "classifier_triplet" => df_tr,
    "M2" => df_m,
    "M2_warmup" => df_mw,
    "self_classifier" => df_self,
    "self_arcface" => df_arc
)

function load_cm(par, modelname, r, full, seed)
    d = filter(:r => ri -> ri == r, filter(:full => fi -> fi == full, par[modelname]))
    nm = savename(d.parameters[1])
    modelpath = datadir("experiments", "MNIST", modelname, "seed=$seed")
    files = readdir(modelpath)
    ixs = findall(f -> (occursin(nm, f) && occursin("r=$r", f) && occursin("full=$full", f)), files)
    accs = map(x -> BSON.load(joinpath(modelpath, files[x]))[:val_acc], ixs)
    CMS = map(x -> BSON.load(joinpath(modelpath, files[x]))[:CM], ixs)
    ix = findmax(accs)[2]
    return CMS[ix]
end

for fl in [true, false]
    for modelname = ["classifier", "classifier_triplet", "self_classifier"]
        for r in [0.002, 0.01, 0.05, 0.1]
            # r = 0.002
            # fl = false
            # modelname = "classifier"
            @info """
            model = $modelname
            r     = $r
            """
            CM = load_cm(par, modelname, r, fl, 5)
            pretty_table(CM[2])
        end
    end
end