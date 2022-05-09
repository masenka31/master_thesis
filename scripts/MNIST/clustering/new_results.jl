using DrWatson
@quickactivate

using Mill, Flux, DataFrames
using master_thesis
using PDMats, Distributions
using ConditionalDists
using PrettyTables

using Plots
ENV["GKSwstype"] = "100"
gr(;markerstrokewidth=0, label="", color=:jet)

df = collect_results(datadir("experiments", "MNIST2"), subfolders=true)

#####################################################
###                 Result tables                 ###
#####################################################

classifier = df[df.modelname .== "classifier", :]
triplet = df[df.modelname .== "classifier_triplet", :]
self = df[df.modelname .== "self_classifier", :]
arcface = df[df.modelname .== "self_arcface", :]
m2 = df[df.modelname .== "M2", :]

modelnames = ["classifier", "classifier_triplet", "M2", "self_classifier", "self_arcface"]
nice_modelnames = ["classifier", "classifier + triplet", "M2 model", "self-supervised classifier", "self-supervised ArcFace"]

res = DataFrame[]
for m in modelnames
    d = df[df.modelname .== m, :].combined_df
    for row in d
        rowf = filter(:method => m -> m != "kNN", row)
        dfirst = first(sort(rowf, :accuracy, rev=true)) |> DataFrame
        res = vcat(res, dfirst[:, [:modelname, :method, :type, :accuracy,:randindex,:MI, :k]])
    end
end
res = vcat(res...)
rvec = repeat([0.05, 0.1, 0.15, 0.2], 5)
res.r = rvec .* 100
sort!(res, :r)
nicenames = repeat(nice_modelnames, 4)
res.modelname = nicenames
# res.type[res.type .== "test_embedding"] .= "test embedding"
# res.type[res.type .== "umap"] .= "train embedding"
res.method[res.method .== "kmeans"] .= "k-means"
res.method[res.method .== "kmedoids"] .= "k-medoids"
res.method[res.method .== "hierarchical_average"] .= "hierarchical (average)"

pretty_table(res, nosubheader=true, formatters = ft_round(3), hlines=[0,1,6,11,16,21], crop=:none)
# this is the input table into master's thesis
pretty_table(res, nosubheader=true, formatters = ft_round(3), hlines=[0,1,6,11,16,21], backend=:latex, tf = tf_latex_booktabs)

################################################
###                 Plotting                 ###
################################################

methods = ["kmeans", "kmedoids", "hierarchical_average", "kNN"]

const markers = [:circle, :square, :utriangle, :dtriangle, :diamond, :hexagon, :star4]
const colorvec = [:blue4, :green4, :darkorange, :purple3, :red3, :grey, :sienna4, :cyan]

function plot_clustering(d, x::Symbol, y::Symbol, knn=true; methods=methods, markers=markers, savename="plot.png", kwargs...)
    mi = d[1, x]
    ma = d[3, x]

    p = plot(;xlabel=x, ylabel=y, kwargs...)
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
        f = filter([:method, :type] => (x, y) -> x == methods[i] && y == "umap", d)
        if i == 1
            p = plot!(f[!, x], f[!, y], color=colorvec[i], ls=:dash, lw=2)
            p = scatter!(f[!, x], f[!, y], marker=markers[i], color=colorvec[i])
        elseif i == 2
            p = plot!(f[!, x], f[!, y], color=colorvec[i], ls=:dash, lw=2, label="UMAP")
            p = scatter!(f[!, x], f[!, y], marker=markers[i], color=colorvec[i], label="k-medoids")
        else
            p = plot!(f[!, x], f[!, y], color=colorvec[i], ls=:dash, lw=2)
            p = scatter!(f[!, x], f[!, y], marker=markers[i], color=colorvec[i], label="hierarchical")
        end
    end

    if knn
        f = filter(:method => x -> x == "kNN", d)
        p = plot!([mi, ma], [f[1, y], f[1, y]], color=colorvec[4], lw=2, label="kNN")
    end
    # savefig("$savename.png")
    savefig(savename)
    return p
end

d = classifier.combined_df[1]
# plot_clustering(d, :k, :accuracy, true, legend=:bottomright)

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
        # wsave(plotsdir("clustering", "MIProblems", "$metric", "plot_r=$(r).png"), p)
        wsave(plotsdir("clustering", "MIProblems2", "$metric", "plot_r=$(r).svg"), p)
    end
end

plot_comparison(:randindex, false)
plot_comparison(:silh, false)
plot_comparison(:accuracy, true)

models = ["classifier", "classifier + triplet", "M2 model", "self-supervised classifier", "self-supervised ArcFace"]
nice_modelnames = ["classifier", "classifier + triplet", "M2 model", "self-supervised classifier", "self-supervised ArcFace"]

function plot_results(table; savename = "plot.png", kwargs...)
    mi = minimum(table.accuracy)
    r = sort(unique(table.r))

    p = plot(;
        legend=:bottomright, ylims=(mi-0.01, 1.005), size=(400, 600), xticks=r,
        xlabel="% of known labels", ylabel="accuracy", labelfontsize=10,
        kwargs...
    )
    for i in 1:length(nice_modelnames)
        t = table[table.modelname .== models[i], :]
        # @show t
        p = plot!(r, t.accuracy, msc = :auto, m=markers[i], label = nice_modelnames[i], ms=5, color=colorvec[i], lw=1.5)
    end
    if isnothing(savename)
        savefig("plot.png")
    else
        # wsave(plotsdir("MIProblems", "$savename.png"), p)
        wsave(plotsdir("MIProblems2", savename), p)
    end
    return p
end

plot_results(res; savename = "cluster_accuracy.png", size=(400,450), ylims=(0.8035555555555554-0.01, 1.005))