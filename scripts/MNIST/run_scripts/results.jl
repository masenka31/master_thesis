using DrWatson
@quickactivate

using DataFrames, Flux, Mill, PrettyTables
using Statistics
using master_thesis
using Distributions, DistributionsAD
using ConditionalDists
using PDMats

df = collect_results!(datadir("experiments", "MNIST", "classifier"), subfolders=true)
df = vcat(df, collect_results!(datadir("experiments", "MNIST", "classifier_triplet"), subfolders=true))
df = vcat(df, collect_results!(datadir("experiments", "MNIST", "M2"), subfolders=true), cols=:union)
df = vcat(df, collect_results!(datadir("experiments", "MNIST", "M2_warmup"), subfolders=true), cols=:union)
df = vcat(df, collect_results!(datadir("experiments", "MNIST", "statistician"), subfolders=true), cols=:union)
self = collect_results!(datadir("experiments", "MNIST", "self_classifier"), subfolders=true)
chamfer = collect_results!(datadir("experiments", "MNIST", "chamfer_knn"), subfolders=true)
# dff = collect_results(datadir("experiments", "MNIST", "ChamferModel"), subfolders=true)

g_model = groupby(df, :modelname)

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

df_c = get_results_table(g_model[1])
df_c2 = hcat(df_c, DataFrame(:modelname => repeat(["classifier"], nrow(df_c))))

df_tr = get_results_table(g_model[2])
df_tr2 = hcat(df_tr, DataFrame(:modelname => repeat(["triplet classifier"], nrow(df_tr))))

df_m = get_results_table(g_model[3])
df_m2 = hcat(df_m, DataFrame(:modelname => repeat(["M2"], nrow(df_m))))

df_mw = get_results_table(g_model[4])
df_mw2 = hcat(df_mw, DataFrame(:modelname => repeat(["M2_warmup"], nrow(df_mw))))

df_s = get_results_table(g_model[5])
df_s2 = hcat(df_s, DataFrame(:modelname => repeat(["statistician"], nrow(df_s))))

df_self = get_results_table(self)
df_self2 = hcat(df_self, DataFrame(:modelname => repeat(["self_classifier"], nrow(df_self))))

df_ch = get_results_chamfer(chamfer)
df_ch2 = hcat(df_ch, DataFrame(:modelname => repeat(["chamfer_kNN"], nrow(df_ch))))
rename!(df_ch2, :k => :parameters)

cls = [:full, :r, :modelname, :train_acc, :val_acc, :test_acc, :parameters]
res = vcat(df_c2[:, cls], df_tr2[:, cls], df_m2[:, cls], df_mw2[:, cls], df_s2[:, cls], df_self2[:, cls], df_ch2[:, cls])

using PrettyTables
pretty_table(res, nosubheader=true, hlines=vcat(0,1,5,9,13,17,21,25,28), formatters = ft_round(3), crop=:none)