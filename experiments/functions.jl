function rosenbrock(x::T) where T<:AbstractVector
    res = 0.0
    for i = 1:size(x,1)-1
        res += 100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
    end
    return res
end

function matfact(x::T, M::Matrix, r::Int) where T<:AbstractVector
    X = zeros(size(M))
    l, n = size(M)

    for i = 1:r
        X .+= x[(i-1)*l+1:(i-1)*l+l]*x[r*l+(i-1)*n:r*l+(i-1)*n+n-1]'
    end

    return (1/2)*LA.norm(M-X)^2
end
