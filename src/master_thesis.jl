module master_thesis

using DrWatson
using Random
using Mill
using StatsBase
using Plots
using Flux
using DistributionsAD, Distributions

include("data.jl")
include("constructors.jl")
include("utils.jl")
include("knn.jl")
include("kl_divergence.jl")
include("confusion_matrix.jl")
include("arcface.jl")

include("models/Models.jl")
# include("gvma/gvma.jl")

export train_test_split
export scatter2, scatter2!, scatter3, scatter3!
export millnet_constructor
export kl_divergence
export safe_softplus
export split_semisupervised_data, split_semisupervised_balanced
export validation_data
export dist_knn, confusion_matrix
export project_data
export reindex, seqids2bags
export arcface_loss, arcface_triplet_loss

end # module