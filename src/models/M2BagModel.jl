# struct M2BagModel
#     prior       # prior - isotropic Gaussian
#     α           # parameters of p(y)
#     bagmodel    # HMill model to encode bag to one vector representation
#     qy_x        # the classification network q(y | x)
#     qz_xy       # encoder q(z | x, y)
#     px_yz       # decoder p(x | z, y)
# end

abstract type M2BagModel end

"""
The original M2 model for bag data.
"""
struct M2Bag <: M2BagModel
    prior       # prior - isotropic Gaussian
    α           # parameters of p(y)
    bagmodel    # HMill model to encode bag to one vector representation
    qy_x        # the classification network q(y | x)
    qz_xy       # encoder q(z | x, y)
    px_yz       # decoder p(x | z, y)
end

"""
The modified M2 model for bag data. The encoder uses information
about individual instances, labels, and a HMill model projection:
q(z | x, y, bagmodel(X)).
"""
struct M2BagDense <: M2BagModel
    prior       # prior - isotropic Gaussian
    α           # parameters of p(y)
    bagmodel    # HMill model to encode bag to one vector representation
    qy_x        # the classification network q(y | x)
    qz_xy       # encoder q(z | x, y, bagmodel(X))
    px_yz       # decoder p(x | z, y)
end

"""
A simpler version of the M2 model for bag data. The instance encoder
only uses the information about instances, but does not get instance labels.
"""
struct M2BagSimple <: M2BagModel
    prior       # prior - isotropic Gaussian
    α           # parameters of p(y)
    bagmodel    # HMill model to encode bag to one vector representation
    qy_x        # the classification network q(y | x)
    qz_xy       # encoder q(z | x)
    px_yz       # decoder p(x | z, y)
end

function Base.show(io::IO, m::M2BagModel)
    s = """$(nameof(typeof(m))):
        prior: N(0, I)
        α: $(softmax(m.α))
        activation: $(match(r"[^)]*(relu|swish|tanh)\)", String(Symbol(m.qz_xy.mapping[1])))[1])
        HMill aggregation: $(typeof(m.bagmodel.layers[1].a))
    """
    print(io, s)
end

Flux.@functor M2Bag
Flux.@functor M2BagDense
Flux.@functor M2BagSimple

function Flux.trainable(m::M2Bag)
    (α=m.α, bagmodel=m.bagmodel, qy_x=m.qy_x, qz_xy=m.qz_xy, px_yz=m.px_yz)
end
function Flux.trainable(m::M2BagDense)
    (α=m.α, bagmodel=m.bagmodel, qy_x=m.qy_x, qz_xy=m.qz_xy, px_yz=m.px_yz)
end
function Flux.trainable(m::M2BagSimple)
    (α=m.α, bagmodel=m.bagmodel, qy_x=m.qy_x, qz_xy=m.qz_xy, px_yz=m.px_yz)
end

function M2_bag_constructor(Xk, c; bdim=2, hdim=4, zdim=2, aggregation=SegmentedMeanMax, activation=swish, type=:vanilla, kwargs...)
    if typeof(activation) == String
        activation = eval(Symbol(activation))
    end
    
    # mill model to get one-vector bag representation
    bagmodel = Chain(reflectinmodel(
        Xk,
        d -> Dense(d, hdim),
        aggregation
    ), Mill.data, Dense(hdim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, bdim), Dense(bdim, c))

    # latent prior - isotropic gaussian
    pz = MvNormal(zeros(Float32, zdim), 1f0)
    # categorical prior
    α = softmax(Float32.(ones(c)))

    # categorical approximate (not used in the current version)
    α_qy_x = Chain(Dense(c,c),softmax)
    qy_x = ConditionalCategorical(α_qy_x)

    # decoder
    xdim = size(Xk[1].data.data, 1)
    net_zx = Chain(Dense(zdim+c,hdim,activation), Dense(hdim, hdim, activation), SplitLayer(hdim, [xdim,xdim], [identity, safe_softplus]))
    px_yz = ConditionalMvNormal(net_zx)

    # encoder
    if type == :vanilla
        net_xz = Chain(Dense(xdim+c,hdim,activation), Dense(hdim, hdim, activation), SplitLayer(hdim, [zdim,zdim], [identity, safe_softplus]))
        qz_xy = ConditionalMvNormal(net_xz)
        return M2Bag(pz, α, bagmodel, qy_x, qz_xy, px_yz)
    elseif type == :dense
        net_xz = Chain(Dense(xdim+c+c,hdim,activation), Dense(hdim, hdim, activation), SplitLayer(hdim, [zdim,zdim], [identity, safe_softplus]))
        qz_xy = ConditionalMvNormal(net_xz)
        return M2BagDense(pz, α, bagmodel, qy_x, qz_xy, px_yz)
    elseif type == :simple
        net_xz = Chain(Dense(xdim,hdim,activation), Dense(hdim, hdim, activation), SplitLayer(hdim, [zdim,zdim], [identity, safe_softplus]))
        qz_xy = ConditionalMvNormal(net_xz)
        return M2BagSimple(pz, α, bagmodel, qy_x, qz_xy, px_yz)
    end
end