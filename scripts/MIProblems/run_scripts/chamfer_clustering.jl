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
include(srcdir("mill_data.jl"))

function try_catch_knn(k, DM, yk, y)
    try
        knn_v, kv = findmax(k -> dist_knn(k, DM, yk, y)[2], 1:k)
        return knn_v, kv
    catch e
        @warn "Caught error, reduced k: $k -> $(k-1)"
        try_catch_knn(k-1, DM, yk, y)
    end
end

function calculate_chamfer(dataset, r, ratios, seed)
    @info "Starting calculation for seed $seed."
    x, y = load_multiclass(dataset)
    data = (data = x, labels = y)

    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.labels; ratios=ratios, seed=seed)
    
    classes = sort(unique(yk))
    n = c = length(classes)
    Xval, yval, Xu, yu = validation_data(yk, Xu, yu, seed, classes)

    xk = map(i -> Xk[i].data.data, 1:nobs(Xk))
    xval = map(i -> Xval[i].data.data, 1:nobs(Xval))
    # xu = map(i -> Xu[i].data.data, 1:nobs(Xu))
    xt = map(i -> Xt[i].data.data, 1:nobs(Xt))

    @info "Data loaded and prepared."

    DM = pairwise(Chamfer(), xval, xk)
    @info "Validation distance matrix calculated."
    val_acc, k = try_catch_knn(30, DM, yk, yval)

    # DMu = pairwise(Chamfer(), xu, xk)
    # @info "Distance matrix for unlabeled data calculated."
    # yunew, unknown_acc = dist_knn(k, DMu, yk, yu)

    DMt = pairwise(Chamfer(), xt, xk)
    @info "Test distance matrix calculated."
    ytnew, test_acc = dist_knn(k, DMt, yk, yt)

    cm, df = confusion_matrix(classes, yt, ytnew)

    results = Dict(
        :modelname => "chamfer_knn",
        :k => k,
        :train_acc => 0,
        :val_acc => val_acc,
        :test_acc => test_acc,
        :CM => (cm, df),
        :seed => seed,
        :r => r,
    )

    @info "Results calculated, saving..."
    nm = savename("k=$k", results, "bson")
    safesave(datadir("experiments", "MIProblems", dataset, "chamfer_knn", "seed=$seed", nm), results)
end

# dataset = "animals"
# for r in [0.05, 0.1, 0.15, 0.2]
#     for seed in 1:15
#         ratios = (r, 0.5-r, 0.5)
#         calculate_chamfer(dataset, r, ratios, seed)
#     end
# end

dataset = "animals_negative"
for r in [0.05, 0.1, 0.15, 0.2]
    for seed in 1:15
        ratios = (r, 0.5-r, 0.5)
        calculate_chamfer(dataset, r, ratios, seed)
    end
end