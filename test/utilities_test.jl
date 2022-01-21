#=
Author: Cooper Simpson

Tests for functionality found in src/utilities.jl -- helpful functionality.
=#

using LinearAlgebra

if run_all || "utilities" in ARGS
    @testset "utilities" begin

        #define the quadratic
        n = 10
        A = randn((n,n))
        f(x) = x'*A*x

        #hvp problem setup
        x = randn(n)
        v = randn(n)
        expected = (A+A')*v

        #=
        Test basic hvp function
        =#
        @testset "basic hvp" begin
            @test CubicNewton._hvp(f, x, v) ≈ expected
        end

        #=
        Test hvp operator
        =#
        @testset "hvp operator" begin
            Hop = CubicNewton.HvpOperator(f, x)
            result = similar(v)
            LinearAlgebra.mul!(result, Hop, v)

            @test eltype(Hop) == eltype(x)
            @test size(Hop) == (n, n)
            @test result ≈ expected
            @test Hop.nProd == 1

            #TODO: Test hessian matrix multiplication, can Dual even handle that?
        end
    end
end
