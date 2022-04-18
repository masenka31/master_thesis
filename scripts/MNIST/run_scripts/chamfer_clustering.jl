using DrWatson
@quickactivate

using master_thesis
using Distances, Clustering
using Flux3D: chamfer_distance

using Distances: UnionMetric
import Distances: result_type

struct Chamfer <: UnionMetric end

(dist::Chamfer)(x, y) = chamfer_distance(x, y)
result_type(dist::Chamfer, x, y) = Float32

include(srcdir("point_cloud.jl"))

data = load_mnist_point_cloud()

function calculate_chamfer(data, r, ratios, full, seed)
    @info "Starting calculation for seed $seed."

    if full
        Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.bag_labels; ratios=ratios, seed=seed)
    else
        # hardcode to only get 4 predefined numbers
        b = map(x -> any(x .== [0,1,3,4]), data.bag_labels)
        filt_data, filt_labels = reindex(data.data, b), data.bag_labels[b]
        Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(filt_data, filt_labels; ratios=ratios, seed=seed)
    end

    classes = sort(unique(yk))
    n = c = length(classes)
    Xval, yval, Xu, yu = validation_data(yk, Xu, yu, seed, classes)

    xk = map(i -> Xk[i].data.data, 1:nobs(Xk))
    xval = map(i -> Xval[i].data.data, 1:nobs(Xval))
    xu = map(i -> Xu[i].data.data, 1:nobs(Xu))
    xt = map(i -> Xt[i].data.data, 1:nobs(Xt))

    @info "Data loaded and prepared."

    DM = pairwise(Chamfer(), xval, xk)
    @info "Validation distance matrix calculated."
    val_acc, k = findmax(k -> dist_knn(k, DM, yk, yval)[2], 1:30)
    parameters = (k = k,)

    # DMu = pairwise(Chamfer(), xu, xk)
    # @info "Distance matrix for unlabeled data calculated."
    # yunew, unknown_acc = dist_knn(k, DMu, yk, yu)

    DMt = pairwise(Chamfer(), xt, xk)
    @info "Test distance matrix calculated."
    ytnew, test_acc = dist_knn(k, DMt, yk, yt)

    cm, df = confusion_matrix(classes, yt, ytnew)

    results = Dict(
        :modelname => "chamfer_knn",
        :parameters => parameters,
        :train_acc => 0,
        :val_acc => val_acc,
        :test_acc => test_acc,
        :CM => (cm, df),
        :seed => seed,
        :r => r,
        :full => full
    )

    @info "Results calculated, saving..."
    nm = savename(savename(parameters), results, "bson")
    safesave(datadir("experiments", "MNIST", "chamfer_knn", "seed=$seed", nm), results)
end

for seed in 1:5
    r, full = 0.002, false
    ratios = (r, 0.5-r, 0.5)
    calculate_chamfer(data, r, ratios, full, seed)
end
for seed in 1:5
    r, full = 0.01, false
    ratios = (r, 0.5-r, 0.5)
    calculate_chamfer(data, r, ratios, full, seed)
end
for seed in 1:5
    r, full = 0.002, true
    ratios = (r, 0.5-r, 0.5)
    calculate_chamfer(data, r, ratios, full, seed)
end
for seed in 1:5
    r, full = 0.01, true
    ratios = (r, 0.5-r, 0.5)
    calculate_chamfer(data, r, ratios, full, seed)
end