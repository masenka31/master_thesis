module master_thesis

using DrWatson
using Random
using Mill
using StatsBase
using Plots
using Flux
using DistributionsAD, Distributions

include(srcdir("data.jl"))
include(srcdir("constructors.jl"))
include(srcdir("utils.jl"))
include(srcdir("knn.jl"))
include(srcdir("kl_divergence.jl"))

export train_test_split
export scatter2, scatter2!, scatter3, scatter3!
export millnet_constructor
export kl_divergence
export safe_softplus
export split_semisupervised_data
export split_semisupervised_balanced

end # module