using DrWatson
@quickactivate

using gvma
using DataFrames, Flux, Mill
using PrettyTables
using Distributions, ConditionalDists

using Plots
ENV["GKSwstype"] = "100"
gr(;markerstrokewidth=0, label="", color=:jet)

df = collect_results(datadir("experiments", "gvma2", "clustering"), subfolders=true)

#####################################################
###                 Result tables                 ###
#####################################################

classifier = df[df.modelname .== "classifier", :]
triplet = df[df.modelname .== "classifier_triplet", :]
self = df[df.modelname .== "self_classifier", :]
arcface = df[df.modelname .== "self_arcface", :]
gen = df[df.modelname .== "genmodel", :]

modelnames = ["classifier", "classifier_triplet", "self_classifier", "self_arcface", "genmodel"]
models = ["classifier", "classifier_triplet", "self_classifier", "self_arcface", "genmodel"]
nice_modelnames = ["classifier", "classifier + triplet", "self-supervised classifier", "self-supervised ArcFace", "M2 model"]

res = DataFrame[]
for m in modelnames
    d = df[df.modelname .== m, :].combined_df
    for row in d
        rowf = filter(:method => m -> m != "kNN", row)
        dfirst = first(sort(rowf, :accuracy, rev=true)) |> DataFrame
        res = vcat(res, dfirst[:, [:modelname, :method, :type, :accuracy, :randindex, :MI, :k]])
        # res = vcat(res, dfirst[:, [:modelname, :method, :type, :accuracy]])
    end
end
res = vcat(res...)
rvec = repeat([0.01, 0.02, 0.05, 0.1, 0.2], 5)
res.r = rvec .* 100
sort!(res, :r)
nicenames = repeat(nice_modelnames, 5)
res.modelname = nicenames
# res.type[res.type .== "test_embedding"] .= "test embedding"
# res.type[res.type .== "train_embedding"] .= "train embedding"
res.method[res.method .== "kmeans"] .= "k-means"
res.method[res.method .== "kmedoids"] .= "k-medoids"
res.method[res.method .== "hierarchical_average"] .= "hierarchical (average)"

pretty_table(res, nosubheader=true, formatters = ft_round(3), hlines=[0,1,6,11,16,21,26], crop=:none)
# this is the input table into master's thesis
pretty_table(res, nosubheader=true, formatters = ft_round(3), hlines=[0,1,6,11,16,21,26], backend=:latex, tf = tf_latex_booktabs)


################################################
###                 Plotting                 ###
################################################

d = classifier.combined_df[1]
methods = ["kmeans", "kmedoids", "hierarchical_average", "kNN"]
# markers = [:square, :circle, :diamond]

const markers = [:circle, :square, :utriangle, :dtriangle, :diamond, :hexagon, :star4]
const colorvec = [:blue4, :green4, :darkorange, :purple3, :red3, :grey, :sienna4, :cyan]

function plot_clustering(d, x::Symbol, y::Symbol, knn=true; methods=methods, markers=markers, savename="plot", kwargs...)
    mi = minimum(skipmissing(d[!, x]))
    ma = maximum(skipmissing(d[!, x]))

    p = plot(;xlabel=x, ylabel=y, kwargs...)#ylims=(0.5, 1.02))
    for i in 1:3
        f = filter([:method, :type] => (x, y) -> x == methods[i] && y == "encoding", d)
        if i == 1
            p = plot!(f[!, x], f[!, y], label="encoding", color=colorvec[i], lw=2)
            p = scatter!(f[!, x], f[!, y], marker=markers[i], label="k-means", color=colorvec[i], lw=2)
        else
            p = plot!(f[!, x], f[!, y], color=colorvec[i], lw=2)
            p = scatter!(f[!, x], f[!, y], marker=markers[i], color=colorvec[i], lw=2)
        end
    end

    for i in 1:3
        f = filter([:method, :type] => (x, y) -> x == methods[i] && y == "train_embedding", d)
        if i == 2
            p = plot!(f[!, x], f[!, y], color=colorvec[i], ls=:dash, lw=2, label="train embedding")
            p = scatter!(f[!, x], f[!, y], marker=markers[i], color=colorvec[i], label="k-medoids")
        else
            p = plot!(f[!, x], f[!, y], color=colorvec[i], ls=:dash, lw=2)
            p = scatter!(f[!, x], f[!, y], marker=markers[i], color=colorvec[i])
        end
    end

    for i in 1:3
        f = filter([:method, :type] => (x, y) -> x == methods[i] && y == "test_embedding", d)
        if i == 3
            p = plot!(f[!, x], f[!, y], color=colorvec[i], ls=:dot, lw=2, label="test embedding")
            p = scatter!(f[!, x], f[!, y], marker=markers[i], color=colorvec[i], label="hierarchical")
        else
            p = plot!(f[!, x], f[!, y], color=colorvec[i], ls=:dot, lw=2)
            p = scatter!(f[!, x], f[!, y], marker=markers[i], color=colorvec[i])
        end
    end
    if knn
        p = plot!([mi, ma], [d[28, y], d[28, y]], color=colorvec[4], lw=2, label="kNN")
        p = plot!([mi, ma], [d[29, y], d[29, y]], color=colorvec[4], lw=2, ls=:dash)
        p = plot!([mi, ma], [d[30, y], d[30, y]], color=colorvec[4], lw=2, ls=:dot)
    end
    # savefig("$savename.png")
    savefig("$savename.svg")
    return p
end

# plot_clustering(d, :k, :accuracy, true, legend=:bottomright)

plot_comparison(:randindex, false)
plot_comparison(:accuracy, true)

metricnames = (
    new_adj_randindex = "adjusted RandIndex", new_randindex = "RandIndex", accuracy = "accuracy",
    adj_randindex = "adjusted RandIndex", randindex = "RandIndex",
)
function plot_comparison(metric, knn)
    for i in 1:4
        r = classifier.r[i]
        ticks = [3,6,9]
        
        d1 = classifier.combined_df[i]
        d2 = triplet.combined_df[i]

        xmin = minimum([minimum(skipmissing(d1[!, metric])), minimum(skipmissing(d2[!, metric]))])
        p1 = plot_clustering(d1, :k, metric, knn; legend=:none, ylabel=metricnames[metric])
        p2 = plot_clustering(d2, :k, metric, knn; legend=:bottomright, ylabel="")
        ri = Int(r*100)
        p = plot(
                p1, p2, layout=(1,2), size=(700,500), title=["Classifier" "Classifier + Triplet"],
                titlefontsize=10, ylims=(xmin-0.05, 1.01), xticks=ticks,
                legendtitle="$ri % labeled, $digits digits", legendtitlefontsize=9,
                foreground_color_legend = nothing, legendfontsize=8
        )
        # wsave(plotsdir("clustering", "gvma", "$metric", "plot_r=$(r).png"), p)
        wsave(plotsdir("clustering", "gvma", "$metric", "plot_r=$(r).svg"), p)
    end
end

function plot_results(table; savename = nothing, kwargs...)
    mi = minimum(table.accuracy)
    r = sort(unique(table.r))

    p = plot(;
        legend=:bottomright, ylims=(mi-0.01, 1.005), size=(400, 600), xticks=r,
        xlabel="% of known labels", ylabel="accuracy", labelfontsize=10,
        kwargs...
    )
    for i in 1:length(nice_modelnames)
        t = table[table.modelname .== nice_modelnames[i], :]
        @show t
        p = plot!(r, t.accuracy, msc = :auto, m=markers[i], label = nice_modelnames[i], ms=5, color=colorvec[i], lw=1.5)
    end
    if isnothing(savename)
        savefig("plot.png")
    else
        # wsave(plotsdir("gvma", "$savename.png"), p)
        wsave(plotsdir("gvma", "$savename.svg"), p)
    end
    return p
end

function plot_metric(table, metric::Symbol; savename = nothing, kwargs...)
    mi = minimum(table[:, metric])
    r = sort(unique(table.r))

    p = plot(;
        legend=:bottomright, ylims=(mi-0.01, 1.005), size=(400, 600), xticks=r,
        xlabel="% of known labels", ylabel="RandIndex", labelfontsize=10,
        kwargs...
    )
    for i in 1:length(nice_modelnames)
        t = table[table.modelname .== nice_modelnames[i], :]
        @show t
        if !ismissing(t[:, metric])
            p = plot!(r, t[:, metric], msc = :auto, m=markers[i], label = nice_modelnames[i], ms=5, color=colorvec[i], lw=1.5)
        end
    end
    if isnothing(savename)
        savefig("plot.png")
    else
        # wsave(plotsdir("gvma", "$savename.png"), p)
        wsave(plotsdir("gvma", "$savename.svg"), p)
    end
    return p
end