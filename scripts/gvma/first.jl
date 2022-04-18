using DrWatson
@quickactivate

using master_thesis
using master_thesis.gvma

include(srcdir("gvma", "init_strain.jl"))

r = 0.01
ratios = (r, 0.5-r, 0.5)
seed = 1

Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(X, y; ratios=ratios, seed=seed)
countmap(yk)
classes = sort(unique(yk))
n = c = length(classes)
Xval, yval, Xu, yu = validation_data(yk, Xu, yu, seed, classes)

dataset = Dataset(
    "/home/maskomic/projects/gvma/data/samples_strain.csv",
    "/home/maskomic/projects/gvma/data/schema.bson",
    "/home/maskomic/projects/gvma/data/samples_strain/"
)
sch = dataset.schema
ex = dataset.extractor

# and encode labels to onehot
Xtrain = Xk
ytrain = yk
yoh_train = Flux.onehotbatch(ytrain, classes)
activation = swish
hdim = 64
aggregation = SegmentedMeanMax

# create a simple classificator model
model = reflectinmodel(
    Xk,
    d -> Dense(d, hdim, activation),
    bag -> SegmentedMeanMax(bag),
    fsm = Dict("" => layer -> Dense(layer, length(classes))),
)
model = Chain(
        mill_model, Mill.data,
        Dense(hdim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, n)
);

opt = ADAM()
loss(x, y) = Flux.logitcrossentropy(Mill.data(model(x)), y)
accuracy(x, y) = round(mean(Flux.onecold(model(x), classes) .== y), digits=5)

batchsize = 64
function minibatch()
    ix = sample(1:nobs(Xk), batchsize)
    xb = reindex(Xk, ix)
    yb = yoh_train[:, ix]
    xb, yb
end

best_model = deepcopy(model);
best_acc = 0

@info "Starting training..."
max_train_time = 60
start_time = time()

while time() - start_time < max_train_time
    batches = map(_ -> minibatch(), 1:1)
    Flux.train!(loss, Flux.params(model), batches, opt)
    Flux.train!(loss, Flux.params(model), zip(xtr, yoh_train), opt)
    acc = accuracy(Xk, yk)
    # @show accuracy(Xk, yk)
    if acc >= best_acc
        # @show accuracy(Xk, yk)
        # @show accuracy(Xval, yval)
        # @show accuracy(Xt, yt)
        best_acc = acc
        best_model = deepcopy(model)
    end
end
@info "Training finished."

xtr = [Xk[i] for i in 1:nobs(Xk)]

opt = ADAM()
loss(x, y) = Flux.logitcrossentropy(Mill.data(model(x)), Flux.onehotbatch(y, classes))
loss(xy::Tuple) = loss(xy...)
minibatches = RandomBatches((xtr, ytrain), size = 64, count = 10)
Flux.Optimise.train!(loss, Flux.params(model), minibatches, ADAM())