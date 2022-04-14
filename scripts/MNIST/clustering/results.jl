using DrWatson
@quickactivate

using DataFrames, Flux, Mill, PDMats
using PrettyTables

df = collect_results(datadir("experiments", "MNIST", "clustering"), subfolders=true)

classifier = df[df.modelname .== "classifier", :]
triplet = df[df.modelname .== "classifier_triplet", :]

for (r1, r2) in zip(eachrow(classifier), eachrow(triplet))
    srt = sort(r1.combined_df, :accuracy, rev=true)
    s1 = srt[1:3, [:method, :type, :accuracy]]

    srt = sort(r2.combined_df, :accuracy, rev=true)
    s2 = srt[1:3, [:method, :type, :accuracy]]

    pretty_table(hcat(s1, s2, makeunique=true), nosubheader=true)
end

# for row in eachrow(triplet)
#     srt = sort(row.combined_df, :accuracy, rev=true)
#     @info srt[1:3, [:method, :accuracy]]
# end

using Plots
gr(label="");
ENV["GKSwstype"] = "100"

d = classifier.combined_df[1]
methods = ["kmeans", "kmedoids", "hierarchical_average", "kNN"]
methods = ["kmeans", "kmedoids", "kNN"]
markers = [:square, :circle, :diamond]

function plot_clustering(d, x::Symbol, y::Symbol; methods=methods, markers=markers, savename="plot")
    mi = minimum(skipmissing(d[!, x]))
    ma = maximum(skipmissing(d[!, x]))

    p = plot(legend=:bottomright, size=(400, 400), xlabel=x, ylabel=y)#ylims=(0.5, 1.02))
    for i in 1:3
        f = filter([:method, :type] => (x, y) -> x == methods[i] && y == "encoding", d)
        p = plot!(f[!, x], f[!, y], marker=markers[i], label=methods[i], color=i, lw=2)
    end
    p = plot!([mi, ma], [d[28, y], d[28, y]], color=4, lw=2, label=methods[4])

    for i in 1:3
        f = filter([:method, :type] => (x, y) -> x == methods[i] && y == "train_embedding", d)
        p = plot!(f[!, x], f[!, y], marker=markers[i], color=i, ls=:dash, lw=2)
    end
    p = plot!([mi, ma], [d[29, y], d[29, y]], color=4, lw=2, ls=:dash)

    for i in 1:3
        f = filter([:method, :type] => (x, y) -> x == methods[i] && y == "test_embedding", d)
        p = plot!(f[!, x], f[!, y], marker=markers[i], color=i, ls=:dot, lw=2)
    end
    p = plot!([mi, ma], [d[30, y], d[30, y]], color=4, lw=2, ls=:dot)
    savefig("$savename.png")
    return p
end

metric = :accuracy
for i in 1:6
    r, f = classifier.r[i], classifier.full[i]
    d1 = classifier.combined_df[i]
    d2 = triplet.combined_df[i]

    xmin = minimum([minimum(d1[!, metric]), minimum(d2[!, metric])])
    p1 = plot_clustering(d1, :k, metric)
    p2 = plot_clustering(d2, :k, metric)
    p = plot(
            p1, p2, layout=(1,2), size=(800,600), title=["Classifier" "Classifier + Triplet"],
            titlefontsize=9, ylims=(xmin-0.05, 1.01)
    )
    # savefig("plot_r=$(r)_full=$f.png")
    wsave(plotsdir("clustering", "$metric", "plot_r=$(r)_full=$f.png"), p)
end
