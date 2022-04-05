using DrWatson
@quickactivate

using Flux, Mill, DataFrames
using Statistics
using ConditionalDists
using Distributions, DistributionsAD
using PDMats
using master_thesis

dfm2 = collect_results(datadir("experiments", "MNIST", "M2"))

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

get_results_M2(dfm2)
pdf = DataFrame(dfm2.parameters)
df = hcat(dfm2, pdf)
g = groupby(df, [:type, :full, :r])
c = combine(g, [:known_acc, :unknown_acc, :test_acc, :chamfer_acc] .=> mean)

s = sort(df, :chamfer_acc, rev=true)