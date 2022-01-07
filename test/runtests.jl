#=
Author: Cooper Simpson

CubicNewton tests, specific tests runnable with Pkg.test(test_args=["target"])
=#
using Test
using CubicNewton

if isempty(ARGS) || "all" in ARGS
    run_all = true
else
    run_all = false
end

include("utilities_test.jl")
include("optimizers_test.jl")
include("flux_test.jl")
