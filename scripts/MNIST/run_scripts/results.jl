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
dff = collect_results(datadir("experiments", "MNIST", "ChamferModel"), subfolders=true)

g_model = groupby(df, :modelname)

function get_results_table(df; k=6)
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
function get_results_M2(df; criterion=:val_acc)
    # group to different experiments
    # full - how many digits, r - how many labels, parameters - controls the model
    # average results over seeds
    f = filter(:val_acc => x -> !ismissing(x), df)
    g = groupby(f, [:full, :r, :parameters])
    @info "Got $(length(g)) groups."

    # filter out experiments that did not run over full 5 seeds
    f = filter(g -> nrow(g) == 5, g)
    @info "$(length(f)) groups left after filtering."

    c = combine(f, [:known_acc, :unknown_acc, :val_acc, :test_acc, :chamfer_known, :chamfer_val, :chamfer_test] .=> mean, renamecols=false)
    par_df = DataFrame(c.parameters)
    # d = sort(hcat(c[:, Not(:parameters)], par_df), :val_acc, rev=true)
    d = sort(hcat(c, par_df), criterion, rev=true)

    gg = groupby(d, [:full, :r])

    # just a check that the right rows are selected
    if sum(map(i -> findmax(gg[i][:, criterion])[2], 1:4) .!= 1) != 0
        error("Check failed, wrong rows would be selected.")
    else
        @info "Check passed. Right rows selected."
    end

    best = mapreduce(i -> DataFrame(gg[i][1, :]), vcat, 1:length(gg))
    sort(best, [:full, :r])
end

df_c = get_results_table(g_model[1])
df_c2 = hcat(df_c, DataFrame(:modelname => repeat(["classifier"], nrow(df_c))))

df_tr = get_results_table(g_model[2])
df_tr2 = hcat(df_tr, DataFrame(:modelname => repeat(["triplet classifier"], nrow(df_tr))))

df_m = get_results_M2(g_model[3])
rename!(df_m, :known_acc => :train_acc)
df_m2 = hcat(df_m, DataFrame(:modelname => repeat(["M2"], nrow(df_m))))

cls = [:full, :r, :modelname, :train_acc, :val_acc, :test_acc, :parameters]
res = vcat(df_c2[:, cls], df_tr2[:, cls], df_m2[:, cls])

using PrettyTables
pretty_table(res, nosubheader=true, hlines=vcat(0,1,7,13,17), formatters = ft_round(3))