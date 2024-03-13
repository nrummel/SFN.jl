#=
Author: Cooper Simpson

Autodiff Hvp benchmarks.
=#

using SFN
using BenchmarkTools
import LinearAlgebra as LA

function rosenbrock(x)
    res = 0.0
    for i = 1:size(x,1)-1
        res += 100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
    end
    return res
end

dim = 100

x = zeros(dim)
result = similar(x)

#Enzyme
Hv = EHvpOperator(rosenbrock, x)
t = @benchmark LA.mul!($result, $Hv, v) setup=(v=randn(dim))
println("Enzyme")
display(t)
println()

#ReverseDiff
Hv = RHvpOperator(rosenbrock, x)
t = @benchmark LA.mul!($result, $Hv, v) setup=(v=randn(dim))
println("ReverseDiff")
display(t)
println()

#Zygote
Hv = ZHvpOperator(rosenbrock, x)
t = @benchmark LA.mul!($result, $Hv, v) setup=(v=randn(dim))
println("Zygote")
display(t)
println()