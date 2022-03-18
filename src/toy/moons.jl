using LinearAlgebra: norm
using StatsBase
using Distributions
using IterTools: product
using Random

"""
    get_obs(x)
Returns bag indices from an of iterable collection.
For a vector of lengths `l = [4,5,2]` returns `obs = [1:4, 5:9, 10:11]`.
"""
function get_obs(x)
    l = nobs.(x)
    n = length(l)
    lv = vcat(0,l)
    mp = map(i -> sum(lv[1:i+1]), 1:n)
    mpv = vcat(0,mp)
    obs = map(i -> mpv[i]+1:mpv[i+1], 1:n)
end

"""
    partition_space(m, space, σ; gap = 1)

Removes values from the discrete space that are in the radius of existing mean.
Takes the mean `m`, current `space` (discrete) and std `σ`. We can also use the
`gap` variable to make the space around the mean bigger.
"""
function partition_space(m, space, σ; gap = 1)
    n = length(m)

    # get only the points which lie in the radius 3σ (plus gap) from the mean
    ss = round(Int, 3.01σ)
    arr = [m[i]-ss-gap:m[i]+ss+gap for i in 1:n]
    _space = collect(product(arr...))
    c = reshape(_space, length(_space))
    b = map(x -> norm(collect(m) - collect(x)), c) .< ss + gap
    get_rid_off = c[b]

    # and remove them from the space
    setdiff(space, get_rid_off)
end

"""
    generate_means(n_classes; σ_noise = 5, σ_code=1, dim = 2, max_val = 100, gap = 1)

For starters, we will only use discrete values for means to make things easier and zero
mean and diagonal variance for the noise distribution.

For N(0,I) the 3σ = 3 => therefore the code means are sampled outside the interval [-4,4].
"""
function generate_means(n_classes; σ_noise = 5, σ_code = 1, dim = 2, max_val = 100, gap = 1)
    _space = collect(product(repeat([-max_val:max_val], dim)...))
    space = reshape(_space, length(_space))
    m = (0, 0)
    space = partition_space(m, space, σ_noise; gap = 2σ_noise)
    means = [m]
    for i in 1:n_classes
        new_mean = sample(space)
        space = partition_space(new_mean, space, σ_code; gap = gap)
        push!(means, new_mean)
    end
    return means, space
end

function generate_means(n_classes, space; σ_code = 1, gap = 1)
    means = []
    for i in 1:n_classes
        new_mean = sample(space)
        space = partition_space(new_mean, space, σ_code; gap = gap)
        push!(means, new_mean)
    end
    return means, space
end

"""
    generate_bag(means; λ = 30, σ_noise = 5, σ_code=1)

Generates a bag from vector of possible means. Samples `n ∼ Po(λ)`
noise points from N(0,σ_noise) and adds one instance from distribution
N(m, σ_code) where `m` is mean sampled from possible means. Returns matrix
with permuted columns.
"""
function generate_bag(means; λ = 30, σ_noise = 5, σ_code = 1)
    # choose mean
    ix = sample(2:length(means))
    m = means[ix]
    # generate code instance
    code = rand(MvNormal(collect(m), σ_code))
    # generate noise
    noise = randn(length(code), rand(Poisson(λ))) .* σ_noise

    # return matrix with permuted columns
    return hcat(code, noise)[:, shuffle(1:size(noise, 2)+1)], ix - 1
end

"""
    generate_data(n_classes, n_bags; max_val = 200, σ_noise = 10, σ_code = 1, gap = 10)

Generates `n_bags` from `n_classes`. Returns data in the form of arrays, labels and instance labels.
"""
function generate_data(n_classes, n_bags; max_val = 200, σ_noise = 10, σ_code = 1, gap = 10)
    m, s = generate_means(n_classes; max_val = max_val, σ_noise = σ_noise, σ_code = σ_code, gap = gap)

    data = []
    instance_labels = []
    labels = []

    for k in 1:n_bags
        bag, lb = generate_bag(m; σ_noise = σ_noise, σ_code = σ_code)
        lb_inst = repeat([lb], size(bag, 2))
        push!(data, bag)
        push!(instance_labels, lb_inst)
        push!(labels, lb)
    end

    l = vcat(instance_labels...)
    return data, labels, l
end

rmat(ϕ) = [cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)]

function moon(n, center, ϕ; sigma = 0.6, radius = 2)
    R = rmat(ϕ)
    noise = rand(n) .* sigma
    theta = pi * rand(n)
    semi_up = hcat((radius .+ noise) .* cos.(theta) .+ center[1], (radius .+ noise) .* sin.(theta) .+ center[2])
    return collect((semi_up * R)')
end

# med = higt'

m, s = generate_means(10)

"""
    generate_mooon_bag(means; λ = 30, sigma = 0.6, radius = 2)

Generates a bag from vector of possible means. Samples `n ∼ Po(λ)`.
"""
function generate_mooon_bag(means, phis; λ = 30, sigma = 0.6, radius = 2)
    # choose mean
    ix = sample(2:length(means))
    m = means[ix]
    ϕ = phis[ix]

    # generate moon
    bag = moon(rand(Poisson(λ)), m, ϕ; sigma = sigma, radius = radius)

    # return matrix with permuted columns
    return bag[:, shuffle(1:size(bag, 2))], ix - 1
end

"""
    generate_data(n_classes, n_bags; max_val = 200, σ_noise = 10, σ_code = 1, gap = 10)

Generates `n_bags` from `n_classes`. Returns data in the form of arrays, labels and instance labels.
"""
function generate_moons_data(n_classes, n_bags; λ=60, max_val = 50, σ_noise = 1, σ_code = 1, gap = 1)
    m, s = generate_means(n_classes; max_val = max_val, σ_noise = σ_noise, σ_code = σ_code, gap = gap)
    phis = rand(Uniform(0, 2π), n_classes+1)

    data = []
    instance_labels = []
    labels = []

    for k in 1:n_bags
        bag, lb = generate_mooon_bag(m, phis; λ=λ, radius = 10, sigma = 4)
        lb_inst = repeat([lb], size(bag, 2))
        push!(data, bag)
        push!(instance_labels, lb_inst)
        push!(labels, lb)
    end

    l = vcat(instance_labels...)
    return data, labels, l
end

#############################################################################################
# other toy datasets #

function twomoon(n; sigma = 0.6, radius = 2, offset = [0.0 0.0])
    center1 = -1.0
    center2 = 1.0

    theta = pi * rand(n)
    noise = rand(n) .* sigma
    semi_up = hcat((radius .+ noise) .* cos.(theta) .+ center1, (radius .+ noise) .* sin.(theta) .- 0.4)
    noise = rand(n) * sigma
    semi_down = hcat((radius .+ noise) .* cos.(-theta) .+ center2, 0.4 .+ (radius .+ noise) .* sin.(-theta))
    x = Matrix(vcat(semi_up .- offset, semi_down .+ offset)')
    y = ones(Int, 2 * n)
    y[n+1:end] .= 2
    x, y
end

function spirals(n, k = 3)
    x = zeros(2, n * k)
    y = zeros(Int, n * k)
    r = range(0.0, stop = 2.5, length = n) |> collect
    for i in 1:k
        t = collect(range((i - 1) * 4, stop = 4 * i, length = n)) + 0.2 * randn(n)
        ix = (i-1)*n+1:i*n
        x[:, ix] = vcat(transpose(r .* sin.(t)), transpose(r .* cos.(t)))
        y[ix] .= i
    end
    (x, y[:])
end

function circles(n, ϵ = 0.1)
    ρ = 2π .* rand(1, 2n)
    y = hcat(fill(1, 1, n), fill(2, 1, n))
    x = [y .* sin.(ρ); y .* cos.(ρ)]
    x = x .+ ϵ .* randn(2, 2n)
    x, y[:]
end