using DrWatson
@quickactivate

using BSON, Mill
using Flux
using Base.Iterators: repeated

corel = BSON.load(datadir("corel_mill.bson"))
@unpack labels, X, bagids = corel

data = BagNode(ArrayNode(X), bagids)

y = []
labelnames = unique(labels)

# create bag labels
for i in 1:2000
    bl = labels[bagids .== i]
    # check that the labels are unique
    if length(unique(bl)) > 1
        @warn "Labels are not the same!"
    end
    push!(y, bl[1])
end

# filter data to have only a few classes
classes = sample(y, 5, replace=false)
b = map(x -> any(x .== classes), y)
data = data[b]
y = y[b]

# Testing on train/test splits

(Xtrain, ytrain), (Xtest, ytest) = train_test_split(data, y, ratio=0.3)
labelnames = unique(y)
c = length(labelnames)
yoh = Flux.onehotbatch(ytrain, labelnames)

loss(x, y) = Flux.logitcrossentropy(full_model(x), y)
accuracy(x, y) = mean(labelnames[Flux.onecold(softmax(full_model(x)))] .== y)

mdim = 3
mill_model = reflectinmodel(
                    Xtrain[1],
                    k -> Dense(k, mdim),
                    d -> meanmax_aggregation(d)
)
full_model = Chain(mill_model, Mill.data, Dense(mdim,c))

ps = Flux.params(full_model)
opt = ADAM()

using Flux: @epochs
@epochs 100 begin
    Flux.train!(loss, ps, repeated((Xtrain, yoh), 50), opt)
    @show accuracy(Xtrain, ytrain)
    @show accuracy(Xtest, ytest)
end

using Plots
gr(markerstrokewidth=0, color=:jet)
using master_thesis: encode

enc = full_model[1:2](Xtrain)
scatter2(enc, zcolor=encode(ytrain, labelnames))
scatter3(enc, zcolor=encode(ytrain, labelnames))

enc_test = full_model[1:2](Xtest)
scatter2(enc_test, zcolor=encode(ytest, labelnames))

using UMAP
using Distances, Clustering

model = UMAP_(enc, 2, n_neighbors=5)
emb = transform(model, enc)
scatter2(emb, zcolor=encode(ytrain, labelnames))

emb_test = transform(model, enc_test)
scatter2(emb_test, zcolor=encode(ytest, labelnames))

# knn
dm = pairwise(Euclidean(), enc_test, enc)
foreach(k -> dist_knn(k, dm, ytrain, ytest), 1:10)

dm_emb = pairwise(Euclidean(), emb_test, emb)
foreach(k -> dist_knn(k, dm_emb, ytrain, ytest), 1:10)