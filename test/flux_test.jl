#=
Author: Cooper Simpson

Tests for functionality found in src/flux.jl -- the CubicNewton interface with
the Flux.jl package.
=#

using Flux
using Zygote

if run_all || "flux" in ARGS
    @testset "flux" begin
        n = 4
        data = randn(n)
        dataBatch = hcat(data, data)
        model = Dense(n, 1, σ)

        #=
        Test basic hessian vector product
        =#
        @testset "hvp" begin
            v = randn(n+1)

            params, re = Flux.destructure(model)
            f(w, x) = re(w)(x)
            f(w) = f(w, dataBatch)

            Hop = CubicNewton.HvpOperator(f, params)
            result = similar(v)
            LinearAlgebra.mul!(result, Hop, v)

            @test result ≈ hessian(w -> sum(f(w, dataBatch)), params)*v

            #TODO: Add some more complicated hvp tests here
        end

        #=
        Test optimizer interface, just making sure everything runs
        =#
        @testset "optimizer" begin
            #TODO: Update this
        end
    end
end
