using DrWatson
@quickactivate

using gvma

include(srcdir("gvma", "init_strain.jl"))

r = 0.01
ratios = (r, 0.5-r, 0.5)
seed = 1

Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(X, y; ratios=ratios, seed=seed)
countmap(yk)
classes = sort(unique(yk))
n = c = length(classes)
Xval, yval, Xu, yu = validation_data(yk, Xu, yu, seed, classes)

# and encode labels to onehot
Xtrain = Xk
xtrain = [Xtrain[i] for i in 1:nobs(Xtrain)]
ytrain = yk
yoh_train = Flux.onehotbatch(ytrain, classes)
yoh_train_vec = [yoh_train[:, i] for i in 1:nobs(Xk)]
activation = swish
hdim = 64
aggregation = SegmentedMeanMax

# create a simple classificator model
mill_model = reflectinmodel(
    # sch, ex,
    Xk[1],
    d -> Dense(d, hdim, activation),
    aggregation
)
model = Chain(
        mill_model, Dense(hdim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, n)
);

opt = ADAM()
loss(x, y) = Flux.logitcrossentropy(model(x), y)
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
max_train_time = 60*5

start_time = time()
while time() - start_time < max_train_time
    batches = map(_ -> minibatch(), 1:2)
    Flux.train!(loss, Flux.params(model), batches, opt)
    acc = accuracy(Xk, yk)
    @show accuracy(Xk, yk)
    if acc >= best_acc
        # @show accuracy(Xk, yk)
        @show accuracy(Xval, yval)
        @show accuracy(Xt, yt)
        best_acc = acc
        best_model = deepcopy(model)
    end
end
@info "Training finished."