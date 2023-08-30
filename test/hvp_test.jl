#=
Author: Cooper Simpson

Tests for functionality found in src/hvp.jl -- helpful functionality.
=#

import LinearAlgebra as LA

if run_all || "hvp" in ARGS
    @testset "hvp" begin

        #define the quadratic
        n = 10
        A = randn((n,n))
        f(x) = LA.dot(x,A,x)

        #hvp problem setup
        x = randn(n)
        v = randn(n)
        single = (A+A')*v
        double = (A+A')*single

        #=
        Test basic hvp function
        =#
        @testset "basic hvp" begin
            @test ehvp(f, x, v) ≈ single
            @test zhvp(f, x, v) ≈ single
            @test rhvp(f, x, v) ≈ single
        end

        #=
        Test hvp operator
        =#
        @testset "hvp operator" begin

            result = similar(v)

            @testset "enzyme" begin
                Hv = EHvpOperator(f, x)
                LA.mul!(result, Hv, v)

                @test eltype(Hv) == eltype(x)
                @test size(Hv) == (n, n)
                @test result ≈ double
                @test Hv.nProd == 2
            end

            @testset "reversediff" begin
                Hv = RHvpOperator(f, x)
                LA.mul!(result, Hv, v)

                @test eltype(Hv) == eltype(x)
                @test size(Hv) == (n, n)
                @test result ≈ double
                @test Hv.nProd == 2
            end

            @testset "zygote" begin
                Hv = ZHvpOperator(f, x)
                LA.mul!(result, Hv, v)

                @test eltype(Hv) == eltype(x)
                @test size(Hv) == (n, n)
                @test result ≈ double
                @test Hv.nProd == 2
            end
        end
    end
end
