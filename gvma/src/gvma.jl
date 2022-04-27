module gvma

using Plots
using StatsBase
using JsonGrinder
using Flux
using Distances
using DrWatson
using Plots

include("dataset.jl")
include("arcface.jl")
include("gen_model.jl")
# include src files from master_thesis
include("/home/maskomic/projects/master_thesis/src/data.jl")
include("/home/maskomic/projects/master_thesis/src/confusion_matrix.jl")
include("/home/maskomic/projects/master_thesis/src/knn.jl")
include("/home/maskomic/projects/master_thesis/src/utils.jl")

import DrWatson: datadir, plotsdir
datadir(args...) = joinpath("/home/maskomic/projects/master_thesis/data", args...)
plotsdir(args...) = joinpath("/home/maskomic/projects/master_thesis/plots", args...)

export Dataset
export datadir, plotsdir
export split_semisupervised_balanced, validation_data, reindex
export confusion_matrix
export dist_knn
export scatter2, scatter2!, scatter3, scatter3!
export arcface_loss, arcface_triplet_loss, arcface_constructor
export LeafModel, loss_enc, model_constructor
export ProbModel, loss_prob, probmodel_constructor
export M2Model, M2constructor, semisupervised_loss

end # module