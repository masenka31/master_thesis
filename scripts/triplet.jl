using DrWatson
@quickactivate

using BSON, Mill
using Flux
using Flux: @epochs
using Base.Iterators: repeated
using ClusterLosses

using Plots
gr(markerstrokewidth=0, color=:jet)
using master_thesis
using master_thesis: encode

using UMAP, Statistics
using Distances, Clustering

corel = BSON.load(datadir("corel_mill.bson"))
@unpack labels, X, bagids = corel

_data = BagNode(ArrayNode(X), bagids)

_y = []
labelnames = unique(labels)

# create bag labels
for i in 1:2000
    bl = labels[bagids .== i]
    # check that the labels are unique
    if length(unique(bl)) > 1
        @warn "Labels are not the same!"
    end
    push!(_y, bl[1])
end

# filter data to have only a few classes
# classes = sample(unique(_y), 10, replace=false)
classes = ["Buses", "Waterfalls", "Desserts"]
b = map(x -> any(x .== classes), _y)
data = _data[b]
y = _y[b]


# Testing on train/test splits
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(data, y, ratio=0.5)
labelnames = unique(y)
c = length(labelnames)
yoh = Flux.onehotbatch(ytrain, labelnames)

# loss with regulatization
function loss_reg(x, yoh, labels, α)
    # cross entropy loss
    ce = Flux.logitcrossentropy(full_model(x), yoh)
    # Jaccard regulatization
    enc = Mill.data(mill_model(x))
    trl = ClusterLosses.loss(Triplet(5f0), SqEuclidean(), enc, labels)

    return ce + α*trl
end
function loss_reg(x, yoh, labels, α)
    # cross entropy loss
    ce = Flux.logitcrossentropy(full_model(x), yoh)
    # triplet regulatization
    enc = mill_model(x)
    trl = ClusterLosses.loss(Triplet(5f0), SqEuclidean(), enc, labels)

    return ce + α*trl
end

α = 1f0
if α == 0
    loss_reg(x, yoh, labels) = Flux.logitcrossentropy(full_model(x), yoh)
else 
    loss_reg(x, yoh, labels) = loss_reg(x, yoh, labels, α)
end

accuracy(x, y) = mean(labelnames[Flux.onecold(full_model(x))] .== y)

##########################################
### Train and save with regularization ###
##########################################

# Parameters are predefined
mdim, activation, nlayers = 10, swish, 2
d = 2
aggregation = SegmentedMeanMax
embdim = 2
emblayers, netlayers = 2, 2
opt = ADAM()

# construct model
seed = 2053
full_model = millnet_constructor(Xtrain, mdim, activation, aggregation, nlayers, odim = c, seed = seed)
full_model = millnet_constructor(Xtrain, mdim, embdim, activation, aggregation, emblayers, netlayers, odim = c, seed = seed)
mill_model = full_model[1:end-netlayers]

# train the model
max_train_time = 60

start_time = time()
@epochs 300 begin
    Flux.train!(loss_reg, Flux.params(full_model), repeated((Xtrain, yoh, ytrain), 10), opt)
    println("train loss: ", loss_reg(Xtrain, yoh, ytrain))
    train_acc = accuracy(Xtrain, ytrain)
    println("accuracy train: ", train_acc)
    println("accuracy test: ", accuracy(Xtest, ytest))
    
    if time() - start_time > max_train_time
        @info "Training time exceeded, stopped training."
        break
    end
end

# encodings
enc = mill_model(Xtrain)#.data
scatter2(enc, zcolor=encode(ytrain, labelnames))

enc_test = mill_model(Xtest)#.data
scatter2(enc_test, zcolor=encode(ytest, labelnames))

# umap embeddings
model = UMAP_(enc, 2, n_neighbors=5)
emb = transform(model, enc)
scatter2(emb, zcolor=encode(ytrain, labelnames))

emb_test = transform(model, enc_test)
scatter2(emb_test, zcolor=encode(ytest, labelnames))

# knn
using master_thesis: dist_knn
dm = pairwise(Euclidean(), enc_test, enc)
foreach(k -> dist_knn(k, dm, ytrain, ytest), 1:10)

dm_emb = pairwise(Euclidean(), emb_test, emb)
foreach(k -> dist_knn(k, dm_emb, ytrain, ytest), 1:10)

###############
## New data ###
###############

# how about new data?
new_classes = setdiff(unique(_y), classes)
b = map(x -> any(x .== new_classes), _y)
newdata = _data[b]
newy = _y[b]

fdata = cat(data, newdata)
fy = vcat(y, newy)

enc = mill_model(fdata)
scatter2(enc, zcolor=encode(fy, unique(_y)),xlims=(-3,7),ylims=(-5,5))

emb = umap(enc, 2, n_neighbors=5)
scatter2(emb, zcolor=encode(fy, unique(_y)))