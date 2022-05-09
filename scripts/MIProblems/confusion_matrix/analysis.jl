using DrWatson
@quickactivate

using master_thesis, StatsBase, BSON
using master_thesis: seqids2bags, reindex, encode
using ThreadTools

# include source files (this file actually calls another file in MNIST scripts folder)
include(srcdir("point_cloud.jl"))
include(srcdir("mill_data.jl"))

# load data tables
include(scriptsdir("MIProblems", "run_scripts", "results_animals.jl"))
par = Dict(
    "classifier" => classifier,
    "classifier_triplet" => triplet,
    "M2" => m2,
    "chamfer_knn" => chamfer,
    "self_classifier" => self,
    "self_arcface" => arc
)
dataset = "animals_negative"

function load_model(par::Dict, dataset::String, modelname::String, r::Number, chosen_seed)
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
    ms = map(f -> BSON.load(f)[:CM], files)
    seeds = map(f -> BSON.load(f)[:seed], files)
    @show seeds
    
    ix = seeds .== chosen_seed
    return ms[ix]
end

models = ["classifier", "classifier_triplet", "M2", "self_classifier", "self_arcface"]
r = 0.1
s = 2
for model in models
    ratios = (r, 0.5-r, 0.5)
    cm, df = load_model(par, dataset, model, r, s)[1]
    pretty_table(cm, nosubheader=true)
end