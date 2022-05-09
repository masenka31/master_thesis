function semisupervised_loss(xk, y, xu, N)
    # known and unknown losses
    l_known = loss_known_bag(xk, y)
    l_unknown = loss_unknown(xu)

    # classification loss on known data
    n = size(project_data(xk), 2)
    lc = 0.1 * N * n * loss_classification(xk, y)

    return l_known + l_unknown + lc
end

loss(xk, yk, xu) = semisupervised_loss(xk, yk, xu, 700)