#=
Author: Cooper Simpson

Tests for functionality found in src/hvp.jl -- helpful functionality.
=#

using LinearAlgebra

if run_all || "hvp" in ARGS
    @testset "hvp" begin

        #define the quadratic
        n = 10
        A = randn((n,n))
        f(x) = x'*A*x

        #hvp problem setup
        x = randn(n)
        v = randn(n)
        single = (A+A')*v
        double = (A+A')*single

        #=
        Test basic hvp function
        =#
        @testset "basic hvp" begin
            @test RSFN._hvp(f, x, v) ≈ single
        end

        #=
        Test hvp operator
        =#
        @testset "hvp operator" begin
            Hop = RSFN.HvpOperator(f, x)
            result = similar(v)
            LinearAlgebra.mul!(result, Hop, v)

            @test eltype(Hop) == eltype(x)
            @test size(Hop) == (n, n)
            @test result ≈ double
            @test Hop.nProd == 2

            #TODO: Test hessian matrix multiplication, can Dual even handle that?
        end
    end
end
