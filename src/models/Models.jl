module Models

using Flux, Mill
using Flux3D: chamfer_distance
using Distributions
using ConditionalDists
using master_thesis

include("M2BagModel.jl")
include("M2bagmodel_loss.jl")
include("ChamferModel.jl")

# M2BagModel
export M2BagModel, M2Bag, M2BagDense, M2BagSimple
export M2_bag_constructor
export semisupervised_loss, semisupervised_loss_Chamfer
export loss_classification_crossentropy
# export reconstruct, reconstruct_mean

# ChamferModel
export ChamferModel
export chamfermodel_constructor
export loss_known, loss_unknown, loss_classification

end