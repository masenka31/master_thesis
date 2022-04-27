using DrWatson
@quickactivate

using gvma
using DataFrames, Flux, Mill, PrettyTables
using Statistics
using JSON, JsonGrinder
using CSV.InlineStrings
using ThreadTools
using ConditionalDists

using Plots
ENV["GKSwstype"] = "100"

# load data into existing collections
thr = Threads.nthreads()
classifier = tmap(seed -> collect_results!(datadir("experiments", "gvma", "classifier", "seed=$seed"), black_list=[:model]), thr, 1:6)
triplet = tmap(seed -> collect_results!(datadir("experiments", "gvma", "classifier_triplet", "seed=$seed"), black_list=[:model]), thr, 1:6)
self = tmap(seed -> collect_results!(datadir("experiments", "gvma", "self_classifier", "seed=$seed"), black_list=[:model]), thr, 1:6)
arcface = tmap(seed -> collect_results!(datadir("experiments", "gvma", "self_arcface", "seed=$seed"), black_list=[:model]), thr, 1:6)
genmodel1 = tmap(seed -> collect_results!(datadir("experiments", "gvma", "_genmodel", "seed=$seed"), black_list=[:model]), thr, 1:6)
genmodel2 = tmap(seed -> collect_results!(datadir("experiments", "gvma", "genmodel", "seed=$seed"), black_list=[:model]), thr, 1:6)
genmodel = vcat(genmodel1, genmodel2)

classifier = filter(:seed => s -> s <= 6, vcat(classifier...))
triplet = filter(:seed => s -> s <= 6, vcat(triplet...))
self = filter(:seed => s -> s <= 6, vcat(self...))
arcface = filter(:seed => s -> s <= 6, vcat(arcface...))
genmodel = filter(:seed => s -> s <= 6, vcat(genmodel...))

function get_results(df; k=6)
    # get the best model based on validation accuracy over 15 seeds
    g = groupby(df, [:r, :parameters])
    f = filter(x -> nrow(x) >= k, g)
    c = combine(f, [:train_acc, :val_acc, :test_acc] .=> mean, [:train_acc, :val_acc, :test_acc] .=> std)
    sort!(c, :val_acc_mean, rev=true)
    g2 = groupby(c, :r)
    d = mapreduce(i -> DataFrame(first(g2[i])), vcat, 1:length(g2))
    sort(d, :r)
end

c = get_results(classifier)
t = get_results(triplet)
s = get_results(self)
a = get_results(arcface)
g = get_results(genmodel)
results = hcat(
    vcat(
        repeat(["classifier"], 5),
        repeat(["triplet classifier"], 5),
        repeat(["self-supervised classifier"], 5),
        repeat(["self-supervised ArcFace"], 5),
        repeat(["M2 model"], 5),
    ),
    vcat(c, t, s, a, g)
)

pretty_table(results[:, Not(:parameters)], nosubheader=true, formatters = ft_round(3), hlines=vcat(0, 1, 6, 11, 16, 21, 26), crop=:none)
pretty_table(sort(results[:, Not(:parameters)], :r), nosubheader=true, formatters = ft_round(3), hlines=vcat(0, 1, 6, 11, 16, 21, 26), crop=:none)

const markers = [:circle, :square, :utriangle, :dtriangle, :diamond, :hexagon, :star4]
const colorvec = [:blue4, :green4, :darkorange, :purple3, :red3, :grey, :sienna4, :cyan]

function plot_results(table; savename = nothing)
    mi = minimum(table.test_acc_mean)
    r = table[1:5, :r] .* 100

    p = plot(
        r, table[1:5, :test_acc_mean], msc = :auto, m=markers[1],
        label = "classifier", ms=5, lw=1, xticks=r,
        yticks = [0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0], color=colorvec[1],
        legend=:bottomright, size=(400, 600),
        xlabel="% of known labels", ylabel="accuracy", labelfontsize=10
    )
    p = plot!(r, table[6:10, :test_acc_mean], msc = :auto, m=markers[2], color=colorvec[2], label = "triplet classifier", ms=5, lw=1)
    p = plot!(r, table[11:15, :test_acc_mean], msc = :auto, m=markers[3], color=colorvec[3], label = "self-supervised classifier", ms=5, lw=1)
    p = plot!(r, table[16:20, :test_acc_mean], msc = :auto, m=markers[4], color=colorvec[4], label = "self-supervised ArcFace", ms=5, lw=1)
    p = plot!(r, table[21:25, :test_acc_mean], msc = :auto, m=markers[5], color=colorvec[5], label = "M2 model", ms=5, lw=1)
    if isnothing(savename)
        savefig("plot.png")
    else
        wsave(plotsdir("gvma", "$savename.png"), p)
    end
    return p
end

# plot_results(results, savename="classifiers2")