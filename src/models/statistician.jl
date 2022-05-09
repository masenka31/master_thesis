"""
    NeuralStatistician(IE, [clength::Int, pc], enc_c, cond_z, enc_z, dec)

Neural Statistician model from https://arxiv.org/abs/1606.02185
paper Towards a Neural Statistician.

Acts on bag data (sets of instances).

# Arguments
* `IE`: instance encoder (trainable neural network)
* `pc`: MvNormal prior p(c)
* `clength`: dimension of context prior MvNormal distribution
* `enc_c`: context encoder q(c|D)
* `cond_z`: conditional p(z|c)
* `enc_z`: instance encoder q(z|x,c)
* `dec`: decoder p(x|z)

# Example
Create a Neural Statistician model:
```julia-repl
julia> idim, vdim, cdim, zdim = 5, 3, 2, 4
julia> instance_enc = Chain(Dense(idim,15,swish),Dense(15,vdim))
julia> enc_c = SplitLayer(vdim,[cdim,1])
julia> enc_c_dist = ConditionalMvNormal(enc_c)
julia> cond_z = SplitLayer(cdim,[zdim,1])
julia> cond_z_dist = ConditionalMvNormal(cond_z)
julia> enc_z = SplitLayer(cdim+vdim,[zdim,1])
julia> enc_z_dist = ConditionalMvNormal(enc_z)
julia> dec = SplitLayer(zdim,[idim,1])
julia> dec_dist = ConditionalMvNormal(dec)
julia> model = NeuralStatistician(instance_enc, cdim, enc_c_dist, cond_z_dist, enc_z_dist, dec_dist)
julia> bag = randn(idim,12)
julia> loss(x) = -elbo(model,x)
julia> loss(bag)
10430.707315113537
```
"""
struct NeuralStatistician{IE,pc <: ContinuousMultivariateDistribution,qc <: ConditionalMvNormal,pz <: ConditionalMvNormal,qz <: ConditionalMvNormal,D <: ConditionalMvNormal} # <: AbstractNS
    instance_encoder::IE
    prior_c::pc
    encoder_c::qc
    conditional_z::pz
    encoder_z::qz
    decoder::D
end

Flux.@functor NeuralStatistician

function Flux.trainable(m::NeuralStatistician)
    (instance_encoder = m.instance_encoder, encoder_c = m.encoder_c, conditional_z = m.conditional_z, encoder_z = m.encoder_z, decoder = m.decoder)
end

function NeuralStatistician(IE, clength::Int, enc_c::ConditionalMvNormal, cond_z::ConditionalMvNormal, enc_z::ConditionalMvNormal, dec::ConditionalMvNormal)
    W = first(Flux.params(enc_c))
    μ_c = fill!(similar(W, clength), 0)
    σ_c = fill!(similar(W, clength), 1)
    prior_c = DistributionsAD.TuringMvNormal(μ_c, σ_c)
    NeuralStatistician(IE, prior_c, enc_c, cond_z, enc_z, dec)
end

"""
    statistician_constructor(Xk, c; hdim::Int, bdim::Int, cdim::Int, zdim::Int, activation="relu", kwargs...)

Constructs basic NeuralStatistician model.

# Arguments
    - `idim::Int`: input dimension
    - `hdim::Int`: size of hidden dimension
    - `vdim::Int`: feature vector dimension
    - `cdim::Int`: context dimension
    - `zdim::Int`: dimension on latent over instances
    - `nlayers::Int=3`: number of layers in model networks, must be >= 3
    - `activation::String="relu"`: activation function
    - `init_seed=nothing`: seed to initialize weights
"""
function statistician_constructor(Xk, c; hdim::Int, bdim::Int, zdim::Int, activation="relu", aggregation="SegmentedMeanMax", kwargs...)
    agg = eval(Symbol(aggregation))
    activation = eval(Symbol(activation))
    
    # instance encoder
	instance_enc = Chain(reflectinmodel(
        Xk,
        d -> Dense(d, hdim),
        BagCount ∘ agg
    ), Mill.data, Dense(hdim, bdim))
	
    # context encoder q(c|D)
	enc_c = Chain(
		Dense(bdim, hdim, activation), Dense(hdim, hdim, activation),
		SplitLayer(hdim, [bdim,bdim], [identity,safe_softplus])
		)
	enc_c_dist = ConditionalMvNormal(enc_c)

    # conditional p(z|c)
	cond_z = Chain(
		Dense(bdim, hdim, activation), Dense(hdim, hdim, activation),
		SplitLayer(hdim, [zdim,zdim], [identity,safe_softplus])
		)
	cond_z_dist = ConditionalMvNormal(cond_z)

    # latent instance encoder q(z|c,x)
	enc_z = Chain(
		Dense(bdim + c, hdim, activation), Dense(hdim, hdim, activation),
		SplitLayer(hdim, [zdim,zdim], [identity,safe_softplus])
		)
	enc_z_dist = ConditionalMvNormal(enc_z)

    # decoder
    xdim = size(Xk[1].data.data, 1)
    dec = Chain(
        Dense(zdim + c, hdim, activation), Dense(hdim, hdim, activation),
        SplitLayer(hdim, [xdim,xdim], [identity,softplus])
    )
    dec_dist = ConditionalMvNormal(dec)

    # get NeuralStatistician model
	model = NeuralStatistician(instance_enc, bdim, enc_c_dist, cond_z_dist, enc_z_dist, dec_dist)
end

using Flux3D: chamfer_distance

"""
    chamfer_elbo(m::NeuralStatistician,x::AbstractArray; β1=1.0, β2=1.0)

Neural Statistician log-likelihood lower bound.
For a Neural Statistician model, simply create a loss
function as
    
    `loss(x) = -elbo(model,x)`

where `model` is a NeuralStatistician type.
The β terms scale the KLDs:
* β1: KL[q(c|D) || p(c)]
* β2: KL[q(z|c,x) || p(z|c)]
"""
function chamfer_loss(m::NeuralStatistician, Xb, y, c; β1=1.0, β2=1.0)
    Xdata = project_data(Xb)
    n = size(Xdata, 2)

    # instance network - bagmodel which outputs one vector
    p = m.instance_encoder(Xb)

    # sample latent for context
    context = rand(condition(m.encoder_c, p))
    yoh = Flux.onehotbatch(y, 1:c)
    cy = vcat(context, yoh)
    h = repeat(cy, outer=(1, n))

    # sample latent for instances
    z = rand(m.encoder_z, h)
    yoh = Flux.onehotbatch(y, 1:c) .* ones(Float32, c, n)
    zy = vcat(z, yoh)
	
    # 3 terms - likelihood, kl1, kl2
    xhat = rand(condition(m.decoder, zy))
    ch = chamfer_distance(Xdata, xhat)

    kld1 = mean(kl_divergence(condition(m.encoder_c, p), m.prior_c))
    kld2 = mean(kl_divergence(condition(m.encoder_z, h), condition(m.conditional_z, context)))

    ch + β1 * kld1 + β2 * kld2
end