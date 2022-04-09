#################################################
###                 Functions                 ###
#################################################

"""
	safe_softplus(x::T)

Safe version of softplus.	
"""
safe_softplus(x::T) where T  = softplus(x) + T(0.000001)

######################################################
###                 Plotting utils                 ###
######################################################

function scatter2(X, x=1, y=2; kwargs...)
    # if size(X,1) > size(X,2)
    #     X = X'
    # end
    scatter(X[x,:],X[y,:]; label="", kwargs...)
    savefig("plot.png")
end
function scatter2!(X, x=1, y=2; kwargs...)
    # if size(X,1) > size(X,2)
    #     X = X'
    # end
    scatter!(X[x,:],X[y,:]; label="", kwargs...)
    savefig("plot.png")
end

function scatter3(X, x=1, y=2, z=3; kwargs...)
    # if size(X,1) > size(X,2)
    #     X = X'
    # end
    scatter(X[x,:],X[y,:],X[z,:]; label="", kwargs...)
end
function scatter3!(X, x=1, y=2, z=3; kwargs...)
    # if size(X,1) > size(X,2)
    #     X = X'
    # end
    scatter!(X[x,:],X[y,:],X[z,:]; label="", kwargs...)
end


# encode labels to numbers
function encode(labels::Vector, labelnames::Vector)
    num_labels = ones(Int, length(labels))
    for i in 1:length(labels)
        v = findall(x -> x == labels[i], labelnames)
        num_labels[i] = v[1]
    end
    return num_labels
end

# encode labels to binary numbers
encode(labels::Vector, missing_class::String) = Int.(labels .== missing_class)

# project data
project_data(X::AbstractBagNode) = Mill.data(Mill.data(X))