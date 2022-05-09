# majority voting clustering

# enc = model(cat(Xk, Xtest))

function new_cluster(enc, clusterfun, ytrain, ytest, k)
    resdf, ycluster = cluster_encoding(enc, ytrain, ytest, clusterfun, k)
    ynew, ynewtest = majority_vote(enc, ytrain, ytest, ycluster, k)
    test_acc = [accuracy(ynewtest, ytest)]
    @info "Test accuracy = $test_acc."
    resdf.accuracy = test_acc
    return resdf
end

function try_catch_cluster(enc, clusterfun, ytrain, ytest, k)
    global iter_max = 0
    try
        return new_cluster(enc, clusterfun, ytrain, ytest, k)
    catch e
        @warn "Failed. Trying again."
        try
            return new_cluster(enc, clusterfun, ytrain, ytest, k)
        catch e
            @warn "Failed. Trying again."
            try
                return new_cluster(enc, clusterfun, ytrain, ytest, k)
            catch e
                @warn "Failed. Trying again."
                try
                    return new_cluster(enc, clusterfun, ytrain, ytest, k)
                catch e
                    @warn "Failed. Trying again."
                    try
                        return new_cluster(enc, clusterfun, ytrain, ytest, k)
                    catch e
                        @warn "Failed. Trying again."
                        return return DataFrame(
                            :method => "$clusterfun",
                            :k => k,
                            :randindex => 0,
                            :adj_randindex => 0,
                            :MI => 0,
                            :silh => 0,
                            :accuracy => 0
                        )
                    end
                end
            end
        end
    end
end



# returns assignments and distance matrix
function cluster_encoding(enc, ytrain, ytest, clusterfun, k)
    DM = pairwise(Euclidean(), enc)
    if clusterfun in [kmeans]
        cbest = assignments(clusterfun(enc, k))
    elseif clusterfun in [kmedoids]
        cbest = assignments(clusterfun(DM, k))
    else
        cbest = clusterfun(DM, k)
    end

    _y = vcat(ytrain, ytest)
    y = encode(_y, unique(_y))

    silh = mean(silhouettes(cbest, DM))
    ri = randindex(y, cbest)
    mi = mutualinfo(y, cbest)
    return DataFrame(
        :method => "$clusterfun",
        :k => k,
        :randindex => ri[2],
        :adj_randindex => ri[1],
        :MI => mi,
        :silh => silh
    ), cbest
end

using Flux, DataFrames

function findmax_label(x)
    typeof(x) <: Dict ? xv = collect(values(x)) : xv = x
    max = maximum(xv)
    mix = findall(a -> a == max, xv)
    if length(mix) == 1
        return findmax(x)
    else
        if typeof(x) <: Dict
            ks = collect(keys(x))
            vals = collect(values(x))
            new_vals = filter(k -> in(k, values(x)), vals[mix])
            new_keys = filter(k -> in(k, keys(x)), ks[mix])
            return map(i -> (new_vals[i], new_keys[i]), 1:length(new_vals))
        else
            return nothing
        end
    end
end

function majority_vote(enc, ytrain, ytest, ycluster, k)
    ntr = length(ytrain)
    nts = length(ytest)
    etr = enc[:, 1:ntr]
    ets = enc[:, ntr+1:end]
    itr = 1:ntr
    its = ntr+1:size(enc,2)

    ### make sure that majority voting is not empty
    # cluster means
    cmeans = map(c -> mean(enc[:, ycluster .== c], dims=2), 1:k)

    cms = []
    ixs = []
    for c in 1:k
        # find all points in cluster c
        ix = ycluster .== c
        # find all train points in cluster c
        ixtr = ix[1:ntr]
        # labels of train points in cluster c
        cm = countmap(ytrain[ixtr])
        push!(cms, cm)
        push!(ixs, ix)
    end

    empty_cm = isempty.(cms)
    not_empty_means = cmeans[.!empty_cm]
    not_empty_labels = (1:k)[.!empty_cm]

    for c in (1:k)[empty_cm]
        m = cmeans[c]
        mins = map(mi -> Flux.mse(mi, m), not_empty_means)
        closest_cluster_ix = findmin(mins)[2]
        ycluster[ixs[c]] .= not_empty_labels[closest_cluster_ix]
    end

    ks = unique(ycluster)
    cms = []
    ixs = []
    for c in ks
        # find all points in cluster c
        ix = ycluster .== c
        # find all train points in cluster c
        ixtr = ix[1:ntr]
        # labels of train points in cluster c
        cm = countmap(ytrain[ixtr])
        push!(cms, cm)
        push!(ixs, ix)
    end

    ### do the majority voting
    clabels = []
    for (c, ix) in zip(cms, ixs)
        l = findmax_label(c)
        push!(clabels, l)
    end

    ynew = Vector{Any}(undef, ntr+nts)

    for (label, ix) in zip(clabels, ixs)
        if !(typeof(label) <: Tuple)
            # for failed voting, choose one label based on silhouettes
            cl = map(c -> c[2], label)
            enctr = enc[:, 1:ntr]
            dm = pairwise(Euclidean(), enctr)
            un = unique(cl)
            s = silhouettes(encode(ytrain, un), dm)
            mean_s = map(c -> mean(s[ytrain .== c]), cl)
            ynew[ix] .= cl[findmin(mean_s)[2]]
        else
            ynew[ix] .= label[2]
        end
    end
    return ynew, ynew[ntr+1:end]
end
