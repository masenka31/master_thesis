using DrWatson
@quickactivate

using DataFrames, Mill, Flux, Statistics
using master_thesis, master_thesis.Models
using PDMats, Distributions, ConditionalDists
using PrettyTables

using Plots
ENV["GKSwstype"] = "100"

function get_results(modelname, dataset)
    # get the best model based on validation accuracy over 15 seeds
    df = collect_results(datadir("experiments", "MIProblems", dataset, modelname), subfolders=true)
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

# animals dataset (datasets not containing the animal)
classifier = get_results("classifier", "animals")
triplet = get_results("classifier_triplet", "animals")
m2 = get_results("M2", "animals")
chamfer = get_chamfer_knn("animals")
cres = vcat(classifier, triplet, m2, chamfer, cols=:union)
modelnames = vcat(
    repeat(["classifier"], 4),
    repeat(["triplet classifier"], 4),
    repeat(["M2 model"], 2),
    repeat(["Chamfer kNN"], 4),
)
cres = hcat(DataFrame(:model => modelnames), cres)

# animals dataset (datasets containing the animal)
classifier2 = get_results("classifier", "animals_negative")
triplet2 = get_results("classifier_triplet", "animals_negative")
m22 = get_results("M2", "animals_negative")
chamfer2 = get_chamfer_knn("animals_negative")
cres2 = vcat(classifier2, triplet2, m22, chamfer2, cols=:union)
cres2 = hcat(DataFrame(:model => modelnames), cres2)

pretty_table(cres, nosubheader=true, formatters = ft_round(3), hlines=vcat(0, 1, 5, 9, 11, 15))
pretty_table(cres2, nosubheader=true, formatters = ft_round(3), hlines=vcat(0, 1, 5, 9, 11, 15))

function plot_results(table; savename = nothing)
    mi = minimum(table.test_acc_mean) - maximum(table.test_acc_std)
    r = table[1:4, :r]

    p = plot(
        r, table[1:4, :test_acc_mean], yerr = table[1:4, :test_acc_std], msc = :auto, m=:circle,
        label = "classifier", ms=5, lw=2,
        legend=:bottomright, ylims=(mi-0.07, 1.02), size=(400, 600),
        xlabel="% of known labels", ylabel="accuracy"
    )
    p = plot!(r, table[5:8, :test_acc_mean], yerr = table[5:8, :test_acc_std], msc = :auto, m=:square, label = "triplet classifier", ms=5, lw=2)
    p = plot!(table[9:11, :r], table[9:11, :test_acc_mean], yerr = table[9:11, :test_acc_std], msc = :auto, m=:diamond, label = "M2 model", ms=5, lw=2)
    p = plot!(r, table[12:15, :test_acc_mean], yerr = table[12:15, :test_acc_std], msc = :auto, m=:utriangle, label = "Chamfer kNN", ms=5, lw=2)
    if isnothing(savename)
        savefig("plot.png")
    else
        wsave(plotsdir("MIProblems", "$savename.png"), p)
    end
    return p
end

# p1 = plot_results(cres; savename = "animals")
# p2 = plot_results(cres2; savename = "animals_negative")

# plot(p1, p2, layout=(1,2), size=(800,600))


# remove files with r=0.25
# for seed in 1:15
#     p = datadir("experiments", "MIProblems", "animals_negative", "M2", "seed=$seed")
#     files = readdir(p)
#     files_filt = filter(f -> occursin("r=0.25", f), files)
#     files = map(f -> joinpath(p, f), files_filt)
#     rm.(files)
# end