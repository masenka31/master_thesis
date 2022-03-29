module Models

using Flux, Mill
using Flux3D: chamfer_distance
using Distributions
using ConditionalDists
using master_thesis

include("M2BagModel.jl")
include("M2bagmodel_loss.jl")

export M2BagModel, M2Bag, M2BagDense, M2BagSimple
export M2_bag_constructor
export semisupervised_loss, semisupervised_loss_Chamfer
export reconstruct, reconstruct_mean

end