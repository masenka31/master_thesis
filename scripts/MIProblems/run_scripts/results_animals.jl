using DrWatson
@quickactivate

using DataFrames, Mill, Flux, Statistics
using master_thesis, master_thesis.Models
using PDMats, Distributions, ConditionalDists
using PrettyTables

using Plots
ENV["GKSwstype"] = "100"

function get_results(modelname, dataset; save=true)
    # get the best model based on validation accuracy over 15 seeds
    if save
        df = collect_results!(datadir("experiments", "MIProblems", dataset, modelname), subfolders=true)
    else
        df = collect_results(datadir("experiments", "MIProblems", dataset, modelname), subfolders=true)
    end
    g = groupby(df, [:r, :parameters])
    f = filter(x -> nrow(x) >= 15, g)
    c = combine(f, [:train_acc, :val_acc, :test_acc] .=> mean, [:train_acc, :val_acc, :test_acc] .=> std)
    # c.r = round.(c.r, digits=2)
    sort!(c, :val_acc_mean, rev=true)
    g2 = groupby(c, :r)
    d = mapreduce(i -> DataFrame(first(g2[i])), vcat, 1:length(g2))
    sort(d, :r)
end
function get_chamfer_knn(dataset)
    df = collect_results(datadir("experiments", "MIProblems", dataset, "chamfer_knn"), subfolders=true)
    g = groupby(df, :r)
    f = filter(x -> nrow(x) >= 15, g)
    c = combine(f, [:train_acc, :val_acc, :test_acc, :k] .=> mean, [:train_acc, :val_acc, :test_acc] .=> std)
    ks = c.k_mean
    c.k_mean = map(x -> (mean_k = x,), ks)
    rename!(c, :k_mean => :parameters)
    sort(c, :r)
end

dataset = "animals_negative"
# animals dataset (datasets containing the animal)
classifier = get_results("classifier", dataset)
triplet = get_results("classifier_triplet", dataset)
m2 = get_results("M2", dataset)
chamfer = get_chamfer_knn(dataset)
self = get_results("self_classifier", dataset)
arc = get_results("self_arcface", dataset)

modelnames = vcat(
    repeat(["classifier"], 4),
    repeat(["classifier + triplet"], 4),
    repeat(["M2 model"], 4),
    repeat(["Chamfer kNN"], 4),
    repeat(["self-classifier"], 4),
    repeat(["self-ArcFace"], 4)
)

cres = vcat(classifier, triplet, m2, chamfer, self, arc, cols=:union)
cres = hcat(DataFrame(:modelname => modelnames), cres)

pretty_table(cres, nosubheader=true, formatters = ft_round(3), hlines=vcat(0, 1, 5, 9, 13, 17, 21, 25), crop=:none)

r = sort(cres[:, Not([:parameters, :train_acc_std, :val_acc_std, :test_acc_std])], :r)
r.r = round.(Int, r.r .* 100)
r.train_acc_mean .= 1
pretty_table(
    r, nosubheader=true, formatters = ft_round(3),
    hlines=[0,1,7,13,19,25], backend=:latex, tf = tf_latex_booktabs
)

function plot_results_yerr(table; savename = nothing)
    mi = minimum(table.test_acc_mean) - maximum(table.test_acc_std)
    r = table[1:4, :r]

    p = plot(
        r, table[1:4, :test_acc_mean], yerr = table[1:4, :test_acc_std], msc = :auto, m=:circle,
        label = "classifier", ms=5, lw=2,
        legend=:bottomright, ylims=(mi-0.07, 1.02), size=(400, 600), xticks=[0.05, 0.1, 0.15, 0.20],
        xlabel="% of known labels", ylabel="accuracy"
    )
    p = plot!(r, table[5:8, :test_acc_mean], yerr = table[5:8, :test_acc_std], msc = :auto, m=:square, label = "triplet classifier", ms=5, lw=2)
    p = plot!(r, table[9:12, :test_acc_mean], yerr = table[9:12, :test_acc_std], msc = :auto, m=:diamond, label = "M2 model", ms=5, lw=2)
    p = plot!(r, table[13:16, :test_acc_mean], yerr = table[13:16, :test_acc_std], msc = :auto, m=:utriangle, label = "Chamfer kNN", ms=5, lw=2)
    p = plot!(r, table[17:20, :test_acc_mean], yerr = table[17:20, :test_acc_std], msc = :auto, m=:dtriangle, label = "self-supervised classifier", ms=5, lw=2)
    if isnothing(savename)
        savefig("plot.png")
    else
        wsave(plotsdir("MIProblems", "$savename.png"), p)
    end
    return p
end

const markers = [:circle, :square, :utriangle, :dtriangle, :diamond, :hexagon, :star4]
const colorvec = [:blue4, :green4, :darkorange, :purple3, :red3, :grey, :sienna4, :cyan]

table = cres

models = ["classifier", "classifier + triplet", "M2 model", "Chamfer kNN", "self-classifier", "self-ArcFace"]
nice_modelnames = ["classifier", "classifier + triplet", "M2 model", "Chamfer kNN", "self-supervised classifier", "self-supervised ArcFace"]

function plot_results(table; savename = nothing, kwargs...)
    mi = minimum(table.test_acc_mean)
    r = table[1:4, :r] .* 100

    p = plot(;
        legend=:bottomright, ylims=(mi-0.01, 1.005), size=(400, 600), xticks=r,
        xlabel="% of known labels", ylabel="accuracy", labelfontsize=10,
        kwargs...
    )
    for i in 1:length(nice_modelnames)
        t = table[table.modelname .== models[i], :]
        #@show t
        p = plot!(r, t.test_acc_mean, msc = :auto, m=markers[i], label = nice_modelnames[i], ms=5, color=colorvec[i], lw=1.5)
    end
    if isnothing(savename)
        savefig("plot.png")
    else
        # wsave(plotsdir("MIProblems", "$savename.png"), p)
        wsave(plotsdir("MIProblems", "$savename.svg"), p)
    end
    return p
end

# plot_results(cres; savename = "results", size=(400, 450))
# plot_results_yerr(cres; savename = "results_yerr")
