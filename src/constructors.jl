"""
    millnet_constructor(Xtrain, mdim, activation, aggregation, nlayers; seed = nothing)

Constructs a classifier as a model composed of Mill model and simple Chain of Dense layers.
The output dimension is fixed to be 10, `mdim` is the hidden dimension in both Mill model
the Chain model.
"""
function millnet_constructor(Xtrain, mdim, activation, aggregation, nlayers; odim = 10, seed = nothing)
    
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    # mill model
    m = reflectinmodel(
        Xtrain[1],
        k -> Dense(k, mdim, activation),
        aggregation
    )

    # create the net after Mill model
    if nlayers == 1
        net = [Dense(mdim, odim)]
    else
        net = vcat(
            repeat([Dense(mdim, mdim, activation)], nlayers-1),
            Dense(mdim, odim)
        )
    end

    # connect the full model
    full_model = Chain(m, Mill.data, net...)

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    # try that the model works
    try
        full_model(Xtrain[1])
    catch e
        error("Model wrong, error: $e")
    end

    return full_model
end

"""
    millnet_constructor(Xtrain, mdim::Int, embdim::Int, activation::Function, aggregation, emblayers::Int, netlayers::Int; odim = 10, seed = nothing)

Constructs a simple model as a concatenation of Mill model, Mill.data, embedding net, and simple Dense net.
"""
function millnet_constructor(Xtrain, mdim::Int, embdim::Int, activation::Function, aggregation, emblayers::Int, netlayers::Int; odim = 10, seed = nothing)
    
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    # mill model
    m = reflectinmodel(
        Xtrain[1],
        k -> Dense(k, mdim, activation),
        aggregation
    )

    # create the ebmedding net
    if emblayers == 1
        embnet = [Dense(mdim, embdim)]
    else
        embnet = vcat(
            repeat([Dense(mdim, mdim, activation)], emblayers-1),
            Dense(mdim, embdim)
        )
    end

    # create the "classification" net
    if netlayers == 1
        net = [Dense(embdim, odim)]
    else
        net = vcat(
            repeat([Dense(embdim, mdim, activation)], netlayers-1),
            Dense(mdim, odim)
        )
    end

    # connect the full model
    full_model = Chain(m, Mill.data, embnet..., net...)

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    # try that the model works
    try
        full_model(Xtrain[1])
    catch e
        error("Model wrong, error: $e")
    end

    return full_model
end