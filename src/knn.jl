"""
    my_findmax(a, ::Colon)

Classic findmax function only returns one label from a dictionary with
maximum value. For kNN, we need all keys with maximum value.
"""
function my_findmax(a, ::Colon)
    p = pairs(a)
    yy = iterate(p)
    col = []
    if yy === nothing
        throw(ArgumentError("collection must be non-empty"))
    end
    # mi - is key
    # m  - is value
    (mi, m), s = yy
    i = mi
    while true
        yy = iterate(p, s)
        yy === nothing && break
        m != m && break
        # i  - key
        # ai - value
        (i, ai), s = yy
        if ai > m # new value is bigger than last saved value
            col = []
            m = ai
            mi = i
            push!(col, (m, mi))
        elseif ai == m
            m = ai
            mi = i
            push!(col, (m, mi))
        end
    end
    if isempty(col)
        return push!(col, (m, mi))
    else
        return col
    end
end
my_findmax(a) = my_findmax(a, :)

"""
    dist_knn(k, distance_matrix, ytrain, ytest)

Given number of neighbors `k` and a distance matrix of size (# test samples, # train samples),
and train and test labels, returns predicted labels and accuracy of predictions.
"""
function dist_knn(k, distance_matrix, ytrain, ytest)
    y_predicted = similar(ytest)
    for i in 1:length(ytest)
        # get indexes of nearest neighbors
        neighbors_ix = partialsortperm(distance_matrix[i,:], 1:k)
        # get labels of nearest neighbors
        neighbors_l = ytrain[neighbors_ix]
        # do majority voting
        c = countmap(neighbors_l)
        mx = my_findmax(c)

        j = k + 1

        length(mx) > 1 ? print("Increasing ") : nothing
        while length(mx) > 1
            print(".")
            neighbors_ix = partialsortperm(distance_matrix[i,:], 1:j)
            neighbors_l = ytrain[neighbors_ix]
            c = countmap(neighbors_l)
            mx = my_findmax(c)
            j += 1
            # println(i, ": ", mx)
        end
        
        # assign the label
        final_label = Tuple(mx...)[2]
        y_predicted[i] = final_label
    end
    acc = mean(y_predicted .== ytest)
    @info "Accuracy: $(round(acc, digits=4))"
    y_predicted, acc
end

dist_matrix(M::AbstractMatrix, train::AbstractArray) = M[length(train)+1:size(M,2), 1:length(train)]
dist_matrix(M::AbstractMatrix, train::AbstractMillNode) = M[nobs(train)+1:size(M,2), 1:nobs(train)]