using Plots
ENV["GKSwstype"] = "100"
gr(label="");

x1 = hcat([0.0,0.1], [.5,0.3])
x2 = hcat([2,2], [2.8,1.9], [2.1,1.7])
Xtrain = hcat(x1, x2)
ytrain = vcat(0,0,1,1,1)

xnew = hcat(
    [0.4,0.7], [0.2, 0.5], [0.25, -0.1], [.6, 0.83], [0.56, 0.7],
    [2.7, 1.1], [2.2, 2.1], [2.3, 1.8],  [2.5, 1.3], [2.6, 1.25], [2.3, 0.8]
)
ynew = vcat(0,0,0,0,0,1,1,1,1,1,1)

scatter(Xtrain[1,:], Xtrain[2,:], color=ytrain, label="labeled")
plot1 = scatter!(xnew[1, :], xnew[2,:], color=4, opacity=0.5, label="unlabeled", legend=:bottomright)

scatter(Xtrain[1,:], Xtrain[2,:], color=ytrain)
scatter!(xnew[1, :], xnew[2,:], color=ynew, opacity=0.5)

xline = -0.3:0.1:3
w1, b1 = [0.70796458 0.61946901], -1.539823011269873
yline1 = @. -(w1[1] / w1[2]) * xline - b1 / w1[2]

w2, b2 = [1.18168906 0.29571363], -1.9545722340299871
yline2 = @. -(w2[1] / w2[2]) * xline - b2 / w2[2]


scatter(Xtrain[1,:], Xtrain[2,:], color=ytrain, ylims=(-0.2, 3), legend=:bottomright)
scatter!(xnew[1, :], xnew[2,:], color=ynew, opacity=0.5)
plot!(xline, yline1, color=:black, label="supervised")
plot2 = plot!(xline, yline2, color=:black, ls=:dash, label="semi-supervised")

plot(plot1, plot2, layout=(1,2),ylims=(-0.2, 2.3), size=(800,400), ms=5)