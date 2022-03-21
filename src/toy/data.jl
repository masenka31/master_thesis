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

# Plotting of the discriminative space

get_rgb(X) = map(xi -> RGBA(probs(condition(qy_x, xi))..., 0), eachcol(X))

function discriminative_space()
    Xgrid = []
    for x in -7.5:0.15:7
        for y in -7.3:0.15:3.5
            push!(Xgrid, [x,y])
        end
    end
    Xgrid = hcat(Xgrid...)
    col = get_rgb(Xgrid);
    scatter2(Xgrid, markerstrokewidth=0, markersize=2.8, marker=:square, color=col, opacity=0.5)
end

function space_plot()
    p = discriminative_space()
    colors = [RGBA(0,1,0,0), RGBA(1,0,0,0), RGBA(0,0,1,0)]
    print(colors)
    for i in 1:3
        p = scatter2!(Xk[:, yk .== i], color=colors[i], ms=6)
        p = scatter2!(Xu[:, yu .== i], color=colors[i], marker=:square, opacity=0.7, ms=2.5)
    end
    return p
end
