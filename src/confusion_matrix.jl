using DataFrames, LinearAlgebra
function confusion_matrix(classes, X, y, predict_labels)
    # get the predicted labels - example
    # function predict_labels(X)
    #     r = probs(condition(qy_x, bagmodel(X)))
    #     ynew = classes[Flux.onecold(r)]
    # end

    ynew = predict_labels(X)

    # calculate the confusion matrix
    n = length(classes)
    C = zeros(Int, n,n)
    for i in 1:n
        for j in 1:n
            b = y .== classes[i]
            bnew = ynew .== classes[j]
            
            bf = b .* bnew
            
            C[i,j] = sum(bf)
        end
    end
    cm = DataFrame(hcat(classes, C), vcat("true \\ predicted", String.(Symbol.(classes))))

    # get the summary DataFrame
    d = diag(C)             # diagonal values (right predictions)
    s = sum(C, dims=2)[:]   # number of predictions for given class
    s2 = sum(C, dims=1)[:]   # number of predictions for given class
    a = d ./ s              # class accuracy (how many did the model get right)

    df = DataFrame(
        :class => classes,
        :accuracy => a,
        :count => Int.(s),
        :predicted => Int.(s2),
        :right => Int.(d)
    )
    df = sort(df, :accuracy)

    return cm, df
end

