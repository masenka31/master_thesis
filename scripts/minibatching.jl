# old minibatch functions

function minibatch(Xk, y, Xu;ksize=64, usize=64)
    kix = sample(1:nobs(Xk), ksize)
    uix = sample(1:nobs(Xu), usize)

    ye = encode(y, classes)
    xk, yk = [Xk[i] for i in kix], ye[kix]
    xu = [Xu[i] for i in uix]

    return xk, yk, xu
end
function minibatch_uniform(Xk, y, Xu;ksize=64, usize=64)
    k = round(Int, ksize / c)
    n = nobs(Xk)

    ik = []
    for i in 1:c
        avail_ix = (1:n)[y .== classes[i]]
        ix = sample(avail_ix, k)
        push!(ik, ix)
    end
    kix = shuffle(vcat(ik...))

    uix = sample(1:nobs(Xu), usize)

    ye = encode(y, classes)
    xk, yk = [Xk[i] for i in kix], ye[kix]
    xu = [Xu[i] for i in uix]

    return xk, yk, xu
end