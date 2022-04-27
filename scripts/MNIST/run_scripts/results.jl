using DrWatson
@quickactivate

using DataFrames, Flux, Mill, PrettyTables
using Statistics
using master_thesis
using Distributions, DistributionsAD
using ConditionalDists
using PDMats

df = collect_results(datadir("experiments", "MNIST", "classifier"), subfolders=true)
df = vcat(df, collect_results(datadir("experiments", "MNIST", "classifier_triplet"), subfolders=true))
df = vcat(df, collect_results(datadir("experiments", "MNIST", "M2"), subfolders=true), cols=:union)
df = vcat(df, collect_results(datadir("experiments", "MNIST", "M2_warmup"), subfolders=true), cols=:union)
self = collect_results(datadir("experiments", "MNIST", "self_classifier"), subfolders=true)
chamfer = collect_results(datadir("experiments", "MNIST", "chamfer_knn"), subfolders=true)
arcface = collect_results(datadir("experiments", "MNIST", "self_arcface"), subfolders=true)

# get full for arcface
c = map(row -> nrow(row.CM[2]), eachrow(arcface))
fl = repeat([true], length(c))
fl[c .== 4] .= false
arcface.full = fl

function get_results_table(df; k=4)
    # group to different experiments
    # full - how many digits, r - how many labels, parameters - controls the model
    # average results over seeds
    g = groupby(df, [:full, :r, :parameters])
    @info "Got $(length(g)) groups."

    # filter out experiments that did not run over full 5 seeds
    f = filter(g -> nrow(g) == 5, g)
    @info "$(length(f)) groups left after filtering."
    c = combine(f, [:train_acc, :val_acc, :test_acc] .=> mean, renamecols=false)
    par_df = DataFrame(c.parameters)
    # d = sort(hcat(c[:, Not(:parameters)], par_df), :val_acc, rev=true)
    d = sort(hcat(c, par_df), :val_acc, rev=true)

    gg = groupby(d, [:full, :r])

    # just a check that the right rows are selected
    if sum(map(i -> findmax(gg[i].val_acc)[2], 1:k) .!= 1) != 0
        error("Check failed, wrong rows would be selected.")
    else
        @info "Check passed. Right rows selected."
    end

    best = mapreduce(i -> DataFrame(gg[i][1, :]), vcat, 1:length(gg))
    sort(best, [:full, :r])
end
function get_results_chamfer(df)
    p = DataFrame(:k => map(p -> p[1], df.parameters))
    df = hcat(df, p)
    g = groupby(df, [:full, :r])
    @info "Got $(length(g)) groups."

    # filter out experiments that did not run over full 5 seeds
    f = filter(g -> nrow(g) == 5, g)
    @info "$(length(f)) groups left after filtering."

    c = combine(f, [:train_acc, :val_acc, :test_acc, :k] .=> mean, renamecols=false)
    sort(c, [:full, :r])
end

g_model = groupby(df, :modelname)

df_c = get_results_table(g_model[1])
df_c2 = hcat(df_c, DataFrame(:modelname => repeat(["classifier"], nrow(df_c))))

df_tr = get_results_table(g_model[2])
df_tr2 = hcat(df_tr, DataFrame(:modelname => repeat(["classifier_triplet"], nrow(df_tr))))

df_m = get_results_table(g_model[3])
df_m2 = hcat(df_m, DataFrame(:modelname => repeat(["M2"], nrow(df_m))))

df_mw = get_results_table(g_model[4])
df_mw2 = hcat(df_mw, DataFrame(:modelname => repeat(["M2_warmup"], nrow(df_mw))))

df_self = get_results_table(self)
df_self2 = hcat(df_self, DataFrame(:modelname => repeat(["self_classifier"], nrow(df_self))))

# df_ch = get_results_chamfer(chamfer)
# df_ch2 = hcat(df_ch, DataFrame(:modelname => repeat(["chamfer_kNN"], nrow(df_ch))))
# rename!(df_ch2, :k => :parameters)

df_arc = get_results_table(arcface)
df_arc2 = hcat(df_arc, DataFrame(:modelname => repeat(["self_arcface"], nrow(df_arc))))

cls = [:full, :r, :modelname, :train_acc, :val_acc, :test_acc, :parameters]
res = vcat(df_c2[:, cls], df_tr2[:, cls], df_m2[:, cls], df_mw2[:, cls], df_self2[:, cls], df_arc2[:, cls])

using PrettyTables
pretty_table(res, nosubheader=true, hlines=vcat(0,1,9,17,25,33,41,49), formatters = ft_round(3), crop=:none)

###########################################################
###                 Nice results tables                 ###
###########################################################

modelnames = ["classifier", "classifier_triplet", "M2", "M2_warmup", "self_classifier", "self_arcface"]
nice_modelnames = ["classifier", "classifier + triplet", "M2", "M2 + warmup", "self-supervised classifier", "self-supervised ArcFace"]

f = filter(:full => f -> f == false, res)
r1 = f[:, [:modelname, :train_acc, :val_acc, :test_acc, :r]]
sort!(r1, :r)
nicenames = repeat(nice_modelnames, 4)
r1.modelname = nicenames

f = filter(:full => f -> f == true, res)
r2 = f[:, [:modelname, :train_acc, :val_acc, :test_acc, :r]]
sort!(r2, :r)
nicenames = repeat(nice_modelnames, 4)
r2.modelname = nicenames

r = hcat(r1, r2[:, Not([:modelname, :r])], makeunique=true)

pretty_table(
    r, nosubheader=true, formatters = ft_round(3),
    hlines=[0,1,7,13,19,25], vlines = [1, 4, 5],
    backend=:latex, tf = tf_latex_booktabs
)

################################################
###                 Plotting                 ###
################################################

modelnames = ["classifier", "classifier_triplet", "M2", "M2_warmup", "self_classifier", "self_arcface"]
nice_modelnames = ["classifier", "classifier + triplet", "M2", "M2 + warmup", "self-supervised classifier", "self-supervised ArcFace"]
markers = [:circle, :square, :utriangle, :ltriangle, :diamond, :hexagon, :star4]

const colorvec = [:blue4, :green4, :darkorange, :purple3, :red3, :grey, :sienna4, :cyan]

function plot_results(table, f; savename = nothing, kwargs...)
    mi = minimum(table.test_acc[table.full .== f])
    r = table[1:4, :r] .* 100

    p = plot(;
        legend=:bottomright, ylims=(mi-0.05, 1.02), size=(400, 600), xticks=r,
        xlabel="% of known labels", ylabel="accuracy", kwargs...
    )
    for i in 1:length(modelnames)
        t = table[(table.modelname .== modelnames[i]) .* (table.full .== f), :]
        # @show t
        p = plot!(r, t.test_acc, msc = :auto, m=markers[i], label = nice_modelnames[i], ms=5, color=colorvec[i], lw=1.5)
    end
    if isnothing(savename)
        savefig("plot.png")
    else
        # wsave(plotsdir("MNIST", "$savename.png"), p)
        wsave(plotsdir("MNIST", "$savename.svg"), p)
    end
    return p
end

# table = res
# pfalse = plot_results(table, false)
# ptrue = plot_results(table, true)

# p = plot(
#     plot_results(table, false; legend=:none),
#     plot_results(table, true; ylabel=""),
#     title=["4 digits" "10 digits"], layout=(1,2), size=(650,500),
#     ylims=(0.5, 1.0), titlefontsize=10, labelfontsize=10
# )
# # savefig("plot.png")
# wsave(plotsdir("MNIST", "classification_accuracy.svg"), p)