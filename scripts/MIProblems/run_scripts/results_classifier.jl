using DrWatson
@quickactivate

using DataFrames, Mill, Flux, Statistics

# get the best model based on validation accuracy over 15 seeds
df = collect_results(datadir("experiments", "MIProblems", "animals", "classifier"), subfolders=true)
g = groupby(df, [:r, :parameters])
f = filter(x -> nrow(x) >= 15, g)
c = combine(f, [:train_acc, :val_acc, :test_acc] .=> mean, [:train_acc, :val_acc, :test_acc] .=> std)
# c.r = round.(c.r, digits=2)
sort!(c, :val_acc_mean, rev=true)
g2 = groupby(c, :r)
d = mapreduce(i -> DataFrame(first(g2[i])), vcat, 1:length(g2))
result_classifier = sort(d, :r)

# get the best model based on validation kNN accuracy over 15 seeds
f = filter(:knn_v => x -> !ismissing(x), df)
g = groupby(f, [:r, :parameters])
f = filter(x -> nrow(x) >= 15, g)
c = combine(f, [:train_acc, :val_acc, :test_acc, :knn_v, :knn_t] .=> mean, [:train_acc, :val_acc, :test_acc, :knn_v, :knn_t] .=> std)
c.r = round.(c.r, digits=2)
sort!(c, :knn_v_mean, rev=true)
g2 = groupby(c, :r)
d = mapreduce(i -> DataFrame(first(g2[i])), vcat, 1:length(g2))
result_classifier_knn = sort(d, :r)

###############
### Triplet ###
###############

df = collect_results(datadir("experiments", "MIProblems", "animals", "classifier_triplet"), subfolders=true)
g = groupby(df, [:r, :parameters])
f = filter(x -> nrow(x) >= 15, g)
c = combine(f, [:train_acc, :val_acc, :test_acc] .=> mean, [:train_acc, :val_acc, :test_acc] .=> std)
sort!(c, :val_acc_mean, rev=true)
g2 = groupby(c, :r)
d = mapreduce(i -> DataFrame(first(g2[i])), vcat, 1:length(g2))
result_classifier_triplet = sort(d, :r)

c = combine(f, [:train_acc, :val_acc, :test_acc, :knn_v, :knn_t] .=> mean, [:train_acc, :val_acc, :test_acc, :knn_v, :knn_t] .=> std)
sort!(c, :knn_v_mean, rev=true)
g2 = groupby(c, :r)
d = mapreduce(i -> DataFrame(first(g2[i])), vcat, 1:length(g2))
result_classifier_triplet_knn = sort(d, :r)

using PrettyTables
pretty_table(result_classifier, nosubheader=true, formatters = ft_round(3))
pretty_table(result_classifier_knn, nosubheader=true, formatters = ft_round(3))
pretty_table(result_classifier_triplet, nosubheader=true, formatters = ft_round(3))
pretty_table(result_classifier_triplet_knn, nosubheader=true, formatters = ft_round(3))

cres = vcat(result_classifier, result_classifier_knn, result_classifier_triplet, result_classifier_triplet_knn, cols=:union)
pretty_table(cres, nosubheader=true, formatters = ft_round(3), hlines=vcat(0, 1, 5, 9, 13, 17))