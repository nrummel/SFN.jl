#=
Author: Cooper Simpson

Tests for functionality found in src/hvp.jl -- helpful functionality.
=#

import LinearAlgebra as LA
import Enzyme, ForwardDiff, ReverseDiff, Zygote

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
            @test RSFN.ehvp(f, x, v) ≈ single
            @test RSFN.zhvp(f, x, v) ≈ single
            @test RSFN.rhvp(f, x, v) ≈ single
        end

        #=
        Test hvp operator
        =#
        @testset "hvp operator" begin

            result = similar(v)

            @testset "enzyme" begin
                Hop = RSFN.EHvpOperator(f, x)
                LA.mul!(result, Hop, v)

                @test eltype(Hop) == eltype(x)
                @test size(Hop) == (n, n)
                @test result ≈ double
                @test Hop.nProd == 2
            end

            @testset "reversediff" begin
                Hop = RSFN.RHvpOperator(f, x)
                LA.mul!(result, Hop, v)

                @test eltype(Hop) == eltype(x)
                @test size(Hop) == (n, n)
                @test result ≈ double
                @test Hop.nProd == 2
            end

            @testset "zygote" begin
                Hop = RSFN.ZHvpOperator(f, x)
                LA.mul!(result, Hop, v)

                @test eltype(Hop) == eltype(x)
                @test size(Hop) == (n, n)
                @test result ≈ double
                @test Hop.nProd == 2
            end
        end
    end
end
