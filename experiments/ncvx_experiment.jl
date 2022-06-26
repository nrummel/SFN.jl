
#=
Author: Cooper Simpson

Non-convex experiment.
=#

using RSFN
import LinearAlgebra as LA
using Random

dim = 10

function michalewicz(x::T) where T<:AbstractVector
    res = 0.0
    for i = 1:size(x,1)
        res += sin(x[i])*sin(i*x[i]^2/pi)^20
    end
    return -res
end

function rosenbrock(x::T) where T<:AbstractVector
    res = 0.0
    for i = 1:size(x,1)-1
        res += 100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
    end
    return res
end

function matfact(x::T) where T<:AbstractVector
    return
end

x = randn(dim)

opt = RSFNOptimizer(size(x, 1))

minimize!(opt, x, rosenbrock, itmax=5000)

exact = ones(dim)

println(LA.norm(exact-x)/LA.norm(exact))
