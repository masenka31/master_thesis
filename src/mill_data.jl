using DataDeps
using DelimitedFiles

# MILL data
#function __init__()
	register(
		DataDep(
			"MIProblems",
			"""
			Dataset: MIProblems
			Authors: Collected by Tomáš Pevný
			Website: https://github.com/pevnak/MIProblems/
			
			Datasets that represent Multiple-Instance problems. 
			""",
			[
				"https://github.com/pevnak/MIProblems/archive/master.zip"
			],
			"9ab2153807d24143d4d0af0b6f4346e349611a4b85d5e31b06d56157b8eed990",
			post_fetch_method = unpack
		))
#end

### MILL data ###
"""
    get_mill_datapath()
Get the absolute path of MIProblems data.
"""
get_mill_datapath() = joinpath(datadep"MIProblems", "MIProblems-master")

"""
	load_mill_data(dataset::String; normalize=true)
Loads basic MIProblems data. For a list of available datasets, do `readdir(GroupAD.get_mill_datapath())`.
"""
function load_mill_data(dataset::String; normalize=true)
	mdp = get_mill_datapath()
	x=readdlm("$mdp/$(dataset)/data.csv",'\t',Float32)
	bagids = readdlm("$mdp/$(dataset)/bagids.csv",'\t',Int)[:]
	y = readdlm("$mdp/$(dataset)/labels.csv",'\t',Int)
	
	# plit to 0/1 classes
	obs_inds0 = vec(y.==0)
	obs_inds1 = vec(y.==1)
	bags0 = seqids2bags(bagids[obs_inds0])
	bags1 = seqids2bags(bagids[obs_inds1])

	# normalize to standard normal
	if normalize 
		x .= standardize(x)
	end
	
	# return normal and anomalous bags
	(normal = BagNode(ArrayNode(x[:,obs_inds0]), bags0), anomaly = BagNode(ArrayNode(x[:,obs_inds1]), bags1),)
end