#=
Author: Cooper Simpson

Tests for functionality found in src/flux.jl -- the CubicNewton interface with
the Flux.jl package.
=#

using Flux

if run_all || "flux" in ARGS
    @testset "flux" begin
        n = 4
        data = randn(n)
        dataBatch = hcat(data, data)
        model = sum(Dense(n, 1, σ))

        #=
        Test basic hessian vector product
        =#
        @testset "hvp" begin
            v = randn(n+1)

            params, re = Flux.destructure(model)
            f(w) = re(w)(dataBatch)

            Hv = CubicNewton.HvpOperator(f, params)
            result = similar(v)
            LinearAlgebra.mul!(result, Hv, v)

            @test result ≈ Flux.hessian(w -> sum(f(w)), params)*v

            #TODO: Add some more complicated hvp tests here
            #TODO: Add gpu tests
        end

        #=
        Test optimizer interface, just making sure everything runs
        =#
        @testset "optimizer" begin
            # model = Flux.Dense(5,5)∘Flux.Dense(5,5)
            # ps, re = Flux.destructure(model)
            # loss(θ, x, y) = Flux.Losses.logitcrossentropy
        end
    end
end
