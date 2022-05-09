"""
    ChamferModel

A model that is used for set reconstruction based on context and Chamfer distance
as a loss function.

Parts:
- `bagmodel` creates a *context* vector from the input set
- `classifier` classifies the bag based on context
- `generator` takes *context* as an input and generates n samples (size of input bag)
- `decoder` transforms generated points to resemble the original bag
"""
struct ChamferModel
    bagmodel
    classifier
    generator
    decoder
end

function Base.show(io::IO, m::ChamferModel)
    s = """$(nameof(typeof(m))):
        bagmodel: $(m.bagmodel)
        classifier: $(m.classifier)
        generator: $(m.generator)
        decoder: $(m.generator)
    """
    print(io, s)
end

Flux.@functor ChamferModel

function Flux.trainable(m::ChamferModel)
    (bagmodel=m.bagmodel, classifier=m.classifier, generator=m.generator, decoder=m.decoder)
end

"""
    chamfermodel_constructor(Xk, c; cdim=2, hdim=4, bdim=4, gdim=4, aggregation=SegmentedMeanMax, activation=swish, kwargs...)

Constructs a ChamferModel.
"""
function chamfermodel_constructor(Xk, c; cdim=2, hdim=4, bdim=4, gdim=4, aggregation=SegmentedMeanMax, activation=swish, kwargs...)
    # get activation function
    if typeof(activation) == String
        activation = eval(Symbol(activation))
    end

    # HMill model
    bagmodel = Chain(reflectinmodel(
        Xk,
        d -> Dense(d, hdim),
        aggregation
    ), Mill.data, Dense(hdim, cdim))

    # classifier
    classifier = Chain(Dense(cdim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, c))

    # generator
    # net_generator = Chain(Dense(cdim+c,hdim,activation), Dense(hdim, hdim, activation), Dense(hdim, hdim, activation), SplitLayer(hdim, [gdim,gdim], [identity, safe_softplus]))
    net_generator = Chain(Dense(cdim,hdim,activation), Dense(hdim, hdim, activation), Dense(hdim, hdim, activation), SplitLayer(hdim, [gdim,gdim], [identity, safe_softplus]))
    generator = ConditionalMvNormal(net_generator)

    # decoder
    xdim = size(Xk[1].data.data, 1)
    decoder = Chain(Dense(gdim+c, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, xdim))

    return ChamferModel(bagmodel, classifier, generator, decoder)
end

function loss_known(m::ChamferModel, Xb, y::Int, c)
    X = project_data(Xb)
    context = m.bagmodel(Xb)
    yoh = Flux.onehot(y, 1:c)
    # context = vcat(x, yoh)

    n = size(X, 2)
    Z = mapreduce(i -> vcat(rand(condition(m.generator, context)), yoh), hcat, 1:n)

    Xhat = m.decoder(Z)

    return chamfer_distance(X, Xhat)
end

function loss_unknown(m::ChamferModel, Xb, c)
    lmat = map(y -> loss_known(m, Xb, y, c), 1:c)
    prob = softmax(m.classifier(m.bagmodel(Xb)))
    l = sum(prob .* lmat)
    e = - entropy(prob)
    return l + e
end

loss_classification(m::ChamferModel, Xb, y, c) = Flux.logitcrossentropy(m.classifier(m.bagmodel(Xb)), Flux.onehotbatch(y, 1:c))