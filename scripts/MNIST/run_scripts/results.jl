using DrWatson
@quickactivate

using DataFrames, Flux, Mill, PrettyTables
using Statistics

df = collect_results(datadir("experiments", "MNIST"), subfolders=true)

g_model = groupby(df, :modelname)

function get_results_table(df)
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
    d = sort(hcat(c[:, Not(:parameters)], par_df), :val_acc, rev=true)

    gg = groupby(d, [:full, :r])

    # just a check that the right rows are selected
    if sum(map(i -> findmax(gg[i].val_acc)[2], 1:6) .!= 1) != 0
        error("Check failed, wrong rows would be selected.")
    else
        @info "Check passed. Right rows selected."
    end

    best = mapreduce(i -> DataFrame(gg[i][1, :]), vcat, 1:length(gg))
    sort(best, [:full, :r])
end

df_c = get_results_table(g_model[1])
df_tr = get_results_table(g_model[2])