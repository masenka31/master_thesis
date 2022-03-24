### MNIST point-cloud ###
# unfortunately this is not available in a direct download format, so we need to do it awkwardly like this
"""
    get_mnist_point_cloud_datapath()

Get the absolute path of the MNIST point cloud dataset. Equals to `datadir("mnist_point_cloud")`.
"""
get_mnist_point_cloud_datapath() = datadir("mnist_point_cloud")

"""
	process_raw_mnist_point_cloud()

One-time processing of MNIST point cloud data that saves them in .bson files.
"""
function process_raw_mnist_point_cloud()
	dp = get_mnist_point_cloud_datapath()

	# check if the path exists
	if !ispath(dp) || length(readdir(dp)) == 0 || !all(map(x->x in readdir(dp), ["test.csv", "train.csv"]))
		mkpath(dp)
		error("MNIST point cloud data are not present. Unfortunately no automated download is available. Please download the `train.csv.zip` and `test.csv.zip` files manually from https://www.kaggle.com/cristiangarcia/pointcloudmnist2d and unzip them in `$(dp)`.")
	end
	
	@info "Processing raw MNIST point cloud data..."
	for fs in ["test", "train"]
		indata = readdlm(joinpath(dp, "$fs.csv"),',',Int32,header=true)[1]
		bag_labels = indata[:,1]
		labels = []
		bagids = []
		data = []
		for (i,row) in enumerate(eachrow(indata))
			# get x data and specify valid values
			x = row[2:3:end]
			valid_inds = x .!= -1
			x = reshape(x[valid_inds],1,:)
			
			# get y and intensity
			y = reshape(row[3:3:end][valid_inds],1,:)
			v = reshape(row[4:3:end][valid_inds],1,:)

			# now append to the lists
			push!(labels, repeat([row[1]], length(x)))
			push!(bagids, repeat([i], length(x)))
			push!(data, vcat(x,y,v))
		end
		outdata = Dict(
			:bag_labels => bag_labels,
			:labels => vcat(labels...),
			:bagids => vcat(bagids...),
			:data => hcat(data...)
			)
		bf = joinpath(dp, "$fs.bson")
		save(bf, outdata)
		@info "Succesfuly processed and saved $bf"
	end
	@info "Done."
end

"""
	load_mnist_point_cloud(;anomaly_class_ind::Int=1 noise=true, normalize=true)

Load the MNIST point cloud data. Anomaly class is chosen as
`anomaly_class = sort(unique(bag_labels))[anomaly_class_ind]`.
"""
function load_mnist_point_cloud(;anomaly_class_ind::Int=1, noise=true, normalize=true)
	dp = get_mnist_point_cloud_datapath()

	# check if the data is there
	if !ispath(dp) || !all(map(x->x in readdir(dp), ["test.bson", "train.bson"]))
		process_raw_mnist_point_cloud()
	end
	
	# load bson data and join them together
	test = load(joinpath(dp, "test.bson"))
	train = load(joinpath(dp, "train.bson"))
	bag_labels = vcat(train[:bag_labels], test[:bag_labels])
	labels = vcat(train[:labels], test[:labels])
	bagids = vcat(train[:bagids], test[:bagids] .+ length(train[:bag_labels]))
	data = Float32.(hcat(train[:data], test[:data]))

	# add uniform noise to dequantize data
	if noise
		data = data .+ rand(Float32, size(data)...)
	end
	
	# choose anomaly class
	anomaly_class = sort(unique(bag_labels))[anomaly_class_ind]
	@info "Loading MNIST point cloud with anomaly class: $(anomaly_class)."

	# split to 0/1 classes - instances
	obs_inds0 = labels .!= anomaly_class
	obs_inds1 = labels .== anomaly_class
	obs0 = seqids2bags(bagids[obs_inds0])
	obs1 = seqids2bags(bagids[obs_inds1])

	# split to 0/1 classes - bags
	bag_inds0 = bag_labels .!= anomaly_class
	bag_inds1 = bag_labels .== anomaly_class	
	l_normal = bag_labels[bag_inds0]
	l_anomaly = bag_labels[bag_inds1]

	# transform data
	if normalize
		data = standardize(data)
	end

	# return normal and anomalous bags (and their labels)
	(normal = BagNode(ArrayNode(data[:,obs_inds0]), obs0), anomaly = BagNode(ArrayNode(data[:,obs_inds1]), obs1), l_normal = l_normal, l_anomaly = l_anomaly)
end

"""
	load_mnist_point_cloud(;anomaly_class_ind::Int=1 noise=true, normalize=true)

Load the MNIST point cloud data. Anomaly class is chosen as
`anomaly_class = sort(unique(bag_labels))[anomaly_class_ind]`.
"""
function load_mnist_point_cloud(;noise=true, normalize=true)
	dp = get_mnist_point_cloud_datapath()

	# check if the data is there
	if !ispath(dp) || !all(map(x->x in readdir(dp), ["test.bson", "train.bson"]))
		process_raw_mnist_point_cloud()
	end
	
	# load bson data and join them together
	test = load(joinpath(dp, "test.bson"))
	train = load(joinpath(dp, "train.bson"))
	bag_labels = vcat(train[:bag_labels], test[:bag_labels])
	labels = vcat(train[:labels], test[:labels])
	bagids = vcat(train[:bagids], test[:bagids] .+ length(train[:bag_labels]))
	data = Float32.(hcat(train[:data], test[:data]))

	# add uniform noise to dequantize data
	if noise
		data = data .+ rand(size(data)...)
	end
    
	# transform data
	if normalize
		data = standardize(data)
	end
	
	obs = seqids2bags(bagids)

	# return normal and anomalous bags (and their labels)
	(data = BagNode(ArrayNode(Float32.(data)), obs), bag_labels = bag_labels, labels = labels)
end

using Statistics
"""
    standardize(Y)

Scales down a 2 dimensional array so it has approx. standard normal distribution. 
Instance = column. 
"""
function StatsBase.standardize(Y::Array{T,2}) where T<:Real
    M, N = size(Y)
    mu = Statistics.mean(Y,dims=2);
    sigma = Statistics.var(Y,dims=2);

    # if there are NaN present, then sigma is zero for a given column -> 
    # the scaled down column is also zero
    # but we treat this more economically by setting the denominator for a given column to one
    # also, we deal with numerical zeroes
    den = sigma
    den[abs.(den) .<= 1e-15] .= 1.0
    den[den .== 0.0] .= 1.0
    den = repeat(sqrt.(den), 1, N)
    nom = Y - repeat(mu, 1, N)
    nom[abs.(nom) .<= 1e-8] .= 0.0
    Y = nom./den
    return Y
end

import Base.length
Base.length(B::BagNode)=length(B.bags.bags)