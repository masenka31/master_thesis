using DrWatson
@quickactivate
using master_thesis
using master_thesis.Models
using Distributions, DistributionsAD
using ConditionalDists
using Flux

using StatsBase, Random
using LinearAlgebra
using Distances
using Base.Iterators: repeated

using Mill

using master_thesis: reindex, seqids2bags, encode

include(srcdir("point_cloud.jl"))

# args
r = parse(Float64, ARGS[1])     # controls ratio of known labels
ratios = (r, 0.5-r, 0.5)        # get the ratios
full = parse(Bool, ARGS[2])

# sample model parameters
function sample_params()
    hdim = sample([8,16,32,64])         # hidden dimension
    ldim = sample([2,4,8,16])           # latent dimension (last layer before softmax layer)
    batchsize = sample([64, 128, 256])
    agg = sample([SegmentedMean, SegmentedMax, SegmentedMeanMax])   # HMill aggregation function
    return hdim, ldim, batchsize, agg
end

# load MNIST data
data = load_mnist_point_cloud()

if full
    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(data.data, data.bag_labels; ratios=ratios, seed=seed)
else
    # hardcode to only get 4 predefined numbers
    b = map(x -> any(x .== [0,1,3,4]), data.bag_labels)
    filt_data, filt_labels = reindex(data.data, b), data.bag_labels[b]
    Xk, yk, Xu, yu, Xt, yt = split_semisupervised_balanced(filt_data, filt_labels; ratios=ratios, seed=seed)
end

# global parameters
classes = sort(unique(yk))
n = c = length(classes)

# model parameters
hdim, ldim, batchsize, agg = sample_params()
parameters = (hdim = hdim, ldim = ldim, batchsize = batchsize, agg = agg)