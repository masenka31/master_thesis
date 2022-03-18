### create the data

using Random
function generate_data(n1, n2, n3; seed=nothing)

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    x1 = randn(2, n1)
    y1 = ones(n1)

    x2 = randn(2, n2) .+ [3.1, -4.2]
    y2 = ones(n2) .+ 1

    x3 = randn(2, n3) .+ [-4.5, -3.7]
    y3 = ones(n3) .+ 2

    N = n1 + n2 + n3
    ix = sample(1:N, N, replace=false)

    X = hcat(x1, x2, x3)[:, ix]
    y = vcat(y1, y2, y3)[ix]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return X, y
end

function split_semisupervised_data(X, y; ratios=(0.3,0.3,0.4), seed=nothing)
    
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    n = length(y)
    ix = sample(1:n, n, replace=false)
    nk, nu, nt = round.(Int, n .* ratios)

    Xk, Xu, Xt = X[:, ix[1:nk]], X[:, ix[nk+1:nk+nu]], X[:, ix[nk+nu+1:n]]
    yk, yu, yt = y[ix[1:nk]], y[ix[nk+1:nk+nu]], y[ix[nk+nu+1:n]]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return Xk, yk, Xu, yu, Xt, yt
end

n1, n2, n3 = 220, 170, 210
X, y = generate_data(n1, n2, n3)
Xk, yk, Xu, yu, Xt, yt = split_semisupervised_data(X, y; ratios=ratios)

### create models

c = 3       # number of classes
dz = 2      # latent dimension
xdim = 2    # input dimension

# latent prior - isotropic gaussian
pz = MvNormal(zeros(dz), 1)

# categorical prior
α = softmax(randn(c))   # α is a trainable parameter!
py = Categorical(α)

# posterior on latent (known labels)
μ_qz_xy = Chain(Dense(xdim+c,2,swish), Dense(2,dz))
σ_qz_xy = Chain(Dense(xdim,2,swish), Dense(2,dz,softplus))
qz_xy(x) = DistributionsAD.TuringDiagMvNormal(μ_qz_xy(x), σ_qz_xy(x[1:xdim]))

# posterior on latent (unknown labels) - not used currently
μ_qz_x = Chain(Dense(xdim,2,swish), Dense(2,dz))
σ_qz_x = Chain(Dense(xdim,2,swish), Dense(2,dz,softplus))
qz_x(x) = DistributionsAD.TuringDiagMvNormal(μ_qz_x(x), σ_qz_x(x))

# posterior on y
α_qy_x = Chain(Dense(2,2,swish), Dense(2,c),softmax)
qy_x(x) = Categorical(α_qy_x(x))

# posterior on x
μ_px_yz = Chain(Dense(xdim+c,2,swish), Dense(2,xdim))
σ_px_yz = Chain(Dense(xdim+c,2,swish), Dense(2,xdim,softplus))
px_yz(x) = DistributionsAD.TuringDiagMvNormal(μ_px_yz(x), σ_px_yz(x))

# parameters and opt
ps = Flux.params(α, μ_qz_xy, σ_qz_xy, μ_qz_x, σ_qz_x, α_qy_x, μ_px_yz, σ_px_yz)
opt = ADAM()