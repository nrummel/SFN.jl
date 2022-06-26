#=
Author: Cooper Simpson

RSFN tests, specific tests runnable with Pkg.test(test_args=["target"])
=#
using Test
using RSFN

if isempty(ARGS) || "all" in ARGS
    run_all = true
else
    run_all = false
end

include("hvp_test.jl")
include("optimizer_test.jl")
include("flux_test.jl")
