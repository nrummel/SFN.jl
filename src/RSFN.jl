#=
Author: Cooper Simpson

CubicNewton optimization package.
=#
module RSFN

#=
Setup
=#
using Requires
using LinearAlgebra

export RSFNOptimizer, minimize!

include("hvp.jl")
include("optimizer.jl")

#=
If optional packages are loaded then export compatible functions.
=#
function __init__()

    @require Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9" begin
        export ehvp, EHvpOperator
        include("hvp/hvp_enzyme.jl")
    end

    @require LinearOperators = "5c8ed15e-5a4c-59e4-a42b-c7e8811fb125" begin
        export LHvpOperator
        include("hvp/hvp_linop.jl")
    end

    @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
            export rhvp, RHvpOperator
            include("hvp/hvp_rdiff.jl")
        end
    end

    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
            export zhvp, ZHvpOperator
            include("hvp/hvp_zygote.jl")
        end
    end

    @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        export StochasticRSFN
        include("flux.jl")
    end
end

end #module
