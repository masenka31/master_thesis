# y are true labels
# ynew are new labels
# classes are sorted class names

n = length(classes)
ynew = Flux.onecold(r, classes)
y = yk

C = zeros(Int, n,n)
for i in 1:n
    for j in 1:n
        b = y .== classes[i]
        bnew = ynew .== classes[j]
        
        bf = b .* bnew
        
        C[i,j] = sum(bf)
    end
end

using DataFrames, PrettyTables, LinearAlgebra
cm = DataFrame(hcat(classes, C), vcat("true \\ predicted", String.(Symbol.(classes))))
pretty_table(cm)

# print accuracies for labels
d = diag(C)
s = sum(C, dims=2)[:]
a = d ./ s

df = DataFrame(hcat(classes, a, s), ["class","accuracy","count"])
sort!(df, :accuracy)

pretty_table(df, crop=:none)

using DataFrames, LinearAlgebra
function confusion_matrix(classes, X, y, best_model)
    # get the predicted labels
    r = probs(condition(qy_x, bagmodel(X)))
    ynew = classes[Flux.onecold(r)]

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

    df = DataFrame(hcat(classes, a, s, s2, d), ["class","accuracy","count","predicted","right"])
    sort!(df, :accuracy)

    return cm, df
end