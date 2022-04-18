using DrWatson
@quickactivate

# include source files (this file actually calls another file in MNIST scripts folder)
include(scriptsdir("MIProblems", "clustering", "cluster_src.jl"))
include(srcdir("point_cloud.jl"))
include(srcdir("mill_data.jl"))

function embeddingfun(modelname, model)
    if modelname in ["classifier", "classifier_triplet"]
        return x -> Mill.data(model[1](x))
    elseif modelname == "M2"
        return x -> Mill.data(model.bagmodel(x))
    end
end

# load data tables
include(scriptsdir("MIProblems", "run_scripts", "results_animals.jl"))

par = Dict("classifier" => classifier, "classifier_triplet" => triplet, "M2" => m2)
r, modelname, dataset = 0.2, "classifier_triplet", "animals"
ratios = (r, 0.5-r, 0.5)
models, seeds, val_accs, test_accs, params = load_models(par, dataset, modelname, r)
R = calculate_clustering_results(models, seeds, val_accs, test_accs, params, dataset, ratios, modelname, r)

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
    :full_df => rdf,
    :combined_df => cdf,
    :parameters => params,
    :r => r,
    :models => models,
    :modelname => modelname
)

safesave(datadir("experiments", "MIProblems", "clustering", dataset, modelname, savename(results_dict, "bson")), results_dict)
sort!(cdf, :accuracy, rev=true)