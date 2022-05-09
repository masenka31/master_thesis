using ClusterLosses, Distances

normalizeW(W::AbstractMatrix) = W ./ sqrt.(sum(W .^2, dims=2))
normalizex(x::AbstractVector) = x ./ sqrt(sum(abs2,x))
normalizex(x::AbstractMatrix) = x ./ sqrt.(sum(x .^ 2, dims=1))
normalizes(x::AbstractVector, s::Real) = x ./ sqrt(sum(abs2,x)) * s
normalizes(x::AbstractMatrix, s::Real) = x ./ sqrt.(sum(x .^ 2, dims=1)) .* s

function arcface_loss(feature_model, W, m, s, x, y, labels)
    x_hat = feature_model(x)
    logit = hardtanh.(normalizeW(W) * normalizex(x_hat))   # this is cos θⱼ
    θ = acos.(logit)                                       # this is θⱼ
    addmargin = cos.(θ .+ m*y)                             # there we get cos θⱼ with added margin to θ(yᵢ)
    scaled = s .* addmargin
    Flux.logitcrossentropy(scaled, y)
end

function arcface_triplet_loss(feature_model, W, m, s, x, y, labels)
    x_hat = feature_model(x)
    logit = hardtanh.(normalizeW(W) * normalizex(x_hat))
    θ = acos.(logit)
    addmargin = cos.(θ .+ m*y)
    scaled = s .* addmargin

    trl = ClusterLosses.loss(Triplet(), SqEuclidean(), x_hat, labels)

    Flux.logitcrossentropy(scaled, y) + trl
end

function arcface_constructor(Xtrain, c; hdim, odim, activation, aggregation, kwargs...)
    aggregation = eval(Symbol(aggregation))
    activation = eval(Symbol(activation))
    mill_model = reflectinmodel(
            Xtrain[1],
            k -> Dense(k, hdim, activation),
            BagCount ∘ aggregation
    )
    feature_model = Chain(mill_model, Dense(hdim, hdim, activation), Dense(hdim, odim))
    W = Flux.kaiming_uniform(c, odim)
    return feature_model, W
end