using DrWatson
@quickactivate

using master_thesis.gvma

using JsonGrinder
using Mill
using Statistics
using Flux
using Flux: throttle, @epochs
using StatsBase
using Base.Iterators: repeated

using Plots
ENV["GKSwstype"] = "100"

function load_dataset(small=false)
    # load strain dataset
    dataset = Dataset(
        # datadir("samples_strain.csv"),
        # datadir("schema.bson"),
        # datadir("samples_strain")
        "/home/maskomic/projects/gvma/data/samples_strain.csv",
        "/home/maskomic/projects/gvma/data/schema.bson",
        "/home/maskomic/projects/gvma/data/samples_strain/"
    )
    X, type, y = dataset[:]
    return X, y, unique(y)
end

X, y, labelnames = load_dataset()
# (X, y, labelnames), (Xs, ys, labelnames_s) = load_dataset(true)

@info "Data loaded and prepared."