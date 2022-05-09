using ConditionalDists, Distributions, DistributionsAD
using Mill.SparseArrays

abstract type AbstractGenerativeModel end

struct LeafModel{S<:Symbol, B<:AbstractMillModel, C<:ConditionalMvNormal, CH<:Chain} <: AbstractGenerativeModel
    key::S
    bagmodel::B
    generator::C
    decoder::CH
    instance_encoder::CH
end

Flux.@functor LeafModel

function Flux.trainable(m::LeafModel)
    (
        bagmodel = m.bagmodel,
        generator = m.generator,
        decoder = m.decoder,
        instance_encoder = m.instance_encoder
    )
end

function Base.show(io::IO, m::LeafModel)
    nm = "LeafModel(:$(m.key))"
	print(io, nm)
end

Base.isempty(x::BagNode) = isempty(x.data.data)

# model = LeafModel(:read_files, bm, gen_dist, decoder, instance_encoder)

_get_float32(x::NGramMatrix) = Matrix{Float32}(sparse(x))
_get_float32(x::MaybeHotMatrix) = Matrix{Float32}(x)
function loss_enc(model::LeafModel, X, y, classes)
    x = Flux.Zygote.@ignore X[model.key]
    if isempty(x)
        return 0f0
    end
    n = Flux.Zygote.@ignore size(x.data)[2]
    m = Flux.Zygote.@ignore _get_float32(x.data.data)
    yoh = Flux.Zygote.@ignore Float32.(repeat(Flux.onehot(y, classes), outer=(1,n)))
    enc = model.instance_encoder(vcat(m, yoh))

    context = vcat(model.bagmodel(x), Flux.onehot(y, classes))
    h = repeat(context, outer=(1, n))
    henc = vcat(h, enc)
    z = rand(condition(model.generator, henc))
    xhat = model.decoder(z)
    
    Flux.mse(m * 100, xhat * 100)
end

function model_constructor(X::ProductNode, key::Symbol, c; hdim=64, cdim=16, zdim=16, activation="swish", aggregation="SegmentedMeanMax")
    # get functions
    activation = eval(Symbol(activation))
    aggregation = eval(Symbol(aggregation))

    # BagModel
    bm = reflectinmodel(
        X[key],
        k -> Dense(k, hdim, activation),
        BagCount ∘ aggregation,
        fsm = Dict("" => hdim -> Dense(hdim, cdim))
    )

    # generating distribution
    idim = size(_get_float32(X[key].data.data), 1)
    gen = Chain(
            Dense(2*cdim+c, hdim, activation), Dense(hdim, hdim, activation),
            SplitLayer(hdim,[zdim,zdim])
    )
    gen_dist = ConditionalMvNormal(gen)

    # decoder and instance encoder
    decoder = Chain(Dense(zdim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, idim, relu))
    instance_encoder = Chain(Dense(idim+c, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, cdim))

    # and return the LeafModel
    LeafModel(key, bm, gen_dist, decoder, instance_encoder)
end

### Probabilistic LeafModel

struct ProbModel{S<:Symbol, B<:AbstractMillModel, C<:ConditionalMvNormal, CH<:Chain} <: AbstractGenerativeModel
    key::S
    bagmodel::B
    generator::C
    decoder::C
    instance_encoder::CH
end

Flux.@functor ProbModel

function Flux.trainable(m::ProbModel)
    (
        bagmodel = m.bagmodel,
        generator = m.generator,
        decoder = m.decoder,
        instance_encoder = m.instance_encoder
    )
end

function Base.show(io::IO, m::ProbModel)
    nm = "ProbModel(:$(m.key))"
	print(io, nm)
end

function probmodel_constructor(X::ProductNode, key::Symbol, c; hdim=64, cdim=16, zdim=16, activation="swish", aggregation="SegmentedMeanMax")
    # get functions
    activation = eval(Symbol(activation))
    aggregation = eval(Symbol(aggregation))

    # BagModel
    bm = reflectinmodel(
        X[key],
        k -> Dense(k, hdim, activation),
        BagCount ∘ aggregation,
        fsm = Dict("" => hdim -> Dense(hdim, cdim))
    )

    # generating distribution
    idim = size(_get_float32(X[key].data.data), 1)
    gen = Chain(
            Dense(2*cdim+c, hdim, activation), Dense(hdim, hdim, activation),
            SplitLayer(hdim,[zdim,zdim],[identity, safe_softplus])
    )
    gen_dist = ConditionalMvNormal(gen)

    # decoder and instance encoder
    dec = Chain(
        Dense(zdim, hdim, activation), Dense(hdim, hdim, activation),
        SplitLayer(hdim,[idim,idim],[identity, safe_softplus])
    )
    dec_dist = ConditionalMvNormal(dec)
    instance_encoder = Chain(Dense(idim+c, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, cdim))

    # and return the ProbModel
    ProbModel(key, bm, gen_dist, dec_dist, instance_encoder)
end

function loss_prob(model::ProbModel, X, y, classes)
    x = Flux.Zygote.@ignore X[model.key]
    if isempty(x)
        return 0f0
    end
    n = Flux.Zygote.@ignore size(x.data)[2]
    m = Flux.Zygote.@ignore _get_float32(x.data.data)
    yoh = Flux.Zygote.@ignore Float32.(repeat(Flux.onehot(y, classes), outer=(1,n)))
    enc = model.instance_encoder(vcat(m, yoh))

    context = vcat(model.bagmodel(x), Flux.onehot(y, classes))
    h = repeat(context, outer=(1, n))
    henc = vcat(h, enc)
    z = rand(model.generator, henc)
    # xhat = model.decoder.μ(z)
    - mean(logpdf(model.decoder, m, z))
end
loss_prob(x, y, classes) = sum(map(model -> loss_prob(model, x, y, classes), model.genmodel))

struct M2Model{T<:Vector{Float32}, S<:Chain, D<:ConditionalDists.AbstractConditionalDistribution, GM<:Tuple}#, GM<:AbstractGenerativeModel}
    α::T
    bagmodel::S
    qy_x::D
    genmodel::GM
end

Flux.@functor M2Model

function Flux.trainable(m::M2Model)
    (
        α = m.α,
        bagmodel = m.bagmodel,
        qy_x = m.qy_x,
        genmodel = m.genmodel
    )
end

function Base.show(io::IO, m::M2Model)
    nm = "M2Model on $(length(m.α)) classes."
	print(io, nm)
end

function M2constructor(X::AbstractMillNode, ks::NTuple, c; hdim=64, cdim=16, zdim=16, bdim=16, activation="swish", aggregation="SegmentedMeanMax", kwargs...)
    gen_models = map(key -> probmodel_constructor(X, key, c; hdim=hdim, cdim=cdim, zdim=zdim, activation=activation, aggregation=aggregation), ks)

    if typeof(activation) == String
        activation = eval(Symbol(activation))
    end
    if typeof(aggregation) == String
        aggregation = eval(Symbol(aggregation))
    end    
    
    # mill model to get one-vector bag representation
    bagmodel = Chain(reflectinmodel(
        X,
        d -> Dense(d, hdim),
        BagCount ∘ aggregation
    ), Dense(hdim, bdim))

    # categorical prior
    α = softmax(Float32.(ones(c)))

    # categorical approximate (not used in the current version)
    α_qy_x = Chain(Dense(bdim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, c), softmax)
    qy_x = ConditionalCategorical(α_qy_x)

    # return the model
    M2Model(α, bagmodel, qy_x, gen_models)
end
function M2constructor(X::AbstractMillNode, Xf::AbstractMillNode, ks::NTuple, c; hdim=64, cdim=16, zdim=16, bdim=16, activation="swish", aggregation="SegmentedMeanMax", kwargs...)
    gen_models = map(key -> probmodel_constructor(X, key, c; hdim=hdim, cdim=cdim, zdim=zdim, activation=activation, aggregation=aggregation), ks)

    if typeof(activation) == String
        activation = eval(Symbol(activation))
    end
    if typeof(aggregation) == String
        aggregation = eval(Symbol(aggregation))
    end    
    
    # mill model to get one-vector bag representation
    bagmodel = Chain(reflectinmodel(
        Xf,
        d -> Dense(d, hdim),
        BagCount ∘ aggregation
    ), Dense(hdim, bdim))

    # categorical prior
    α = softmax(Float32.(ones(c)))

    # categorical approximate (not used in the current version)
    α_qy_x = Chain(Dense(bdim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, c), softmax)
    qy_x = ConditionalCategorical(α_qy_x)

    # return the model
    M2Model(α, bagmodel, qy_x, gen_models)
end

function loss_known(model::M2Model, X, y, c)
    # reconstruction loss
    llh = sum(map(model -> loss_prob(model, X, y, 1:c), model.genmodel))

    # p(y)
    lpy = -logpdf(Categorical(softmax(model.α)), y)

    # kl?
    return llh + lpy
end

function loss_unknown(model::M2Model, X, c)
    l = map(y -> loss_known(model, X, y, c), 1:c)
    prob = condition(model.qy_x, model.bagmodel(X)).α

    l = sum(prob .* l)
    e = - mean(entropy(condition(model.qy_x, model.bagmodel(X))))
    # e = - entropy(prob)
    return l + e
end
function loss_unknown(model::M2Model, X, Xf, c)
    l = map(y -> loss_known(model, X, y, c), 1:c)
    prob = condition(model.qy_x, model.bagmodel(Xf)).α

    l = sum(prob .* l)
    e = - mean(entropy(condition(model.qy_x, model.bagmodel(Xf))))
    # e = - entropy(prob)
    return l + e
end

loss_classification(model::M2Model, X, y, c) = Flux.crossentropy(condition(model.qy_x, model.bagmodel(X)).α, Flux.onehotbatch(y, 1:c))

function semisupervised_loss(model::M2Model, xk, y, xu, c, N)
    # known and unknown losses
    l_known = loss_known(model, xk, y, c)
    l_unknown = loss_unknown(model, xu, c)

    # classification loss on known data
    # lc = N * loss_classification(model, xk, y)
    lc = N * loss_classification(model, xk, y, c)

    return l_known + l_unknown + lc
end
function semisupervised_loss(model::M2Model, xk, xf, y, xu, c, N)
    # known and unknown losses
    l_known = loss_known(model, xk, y, c)
    l_unknown = loss_unknown(model, xu, xf, c)

    # classification loss on known data
    # lc = N * loss_classification(model, xk, y)
    lc = N * loss_classification(model, xf, y, c)

    return l_known + l_unknown + lc
end