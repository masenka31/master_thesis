using DrWatson
@quickactivate

using gvma
include(srcdir("init_strain.jl"))

using DataFrames, Flux, Mill
using Distributions, ConditionalDists

include(scriptsdir("run_scripts", "results.jl"))
include(scriptsdir("clustering", "cluster_src.jl"))

par = Dict(
    "classifier" => c,
    "classifier_triplet" => t,
    "self_classifier" => s,
    "self_arcface" => a,
    "genmodel" => g
)

function embeddingfun(modelname, model)
    if modelname in ["classifier", "classifier_triplet", "self_classifier"]
        return x -> model[1](x)
    elseif modelname == "genmodel"
        return x -> model.bagmodel(x)
    elseif modelname == "self_arcface"
        return x -> model(x)
    end
end

modelname = ARGS[1]
r = parse(Float64, ARGS[2])
ratios = (r, 0.5-r, 0.5)
models, seeds, val_accs, test_accs, params = load_models(par, modelname, r);

Xf = X[:behavior_summary]

for r in [0.01, 0.02, 0.05, 0.1, 0.2] #[0.01, 0.02, 0.05, 0.1, 0.2]
    @info "Calculating clustering for r=$r for $modelname."
    ratios = (r, 0.5-r, 0.5)
    models, seeds, val_accs, test_accs, params = load_models(par, modelname, r);
    R = calculate_clustering_results(models, seeds, val_accs, test_accs, params, Xf, y, ratios, modelname, r)

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

    safesave(datadir("experiments", "gvma", "clustering", modelname, savename(results_dict, "bson")), results_dict)
    @info "Results for ratio $r saved."
end