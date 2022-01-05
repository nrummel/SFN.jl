#=
Author: Cooper Simpson

CubicNewton tests
=#
using Test
using CubicNewton
using Flux

#=
Test cubic newton compatability with Flux.
=#
@testset "flux" begin
    struct Quadratic{T<:AbstractArray}
		w::T
	end

	Flux.@functor Quadratic

	function (f::Quadratic)(x::AbstractArray)
		return f.w'*(x*x')*f.w
	end

    n = rand(5:10)
    data = randn(n) #TODO: convert to use batches
    model = Quadratic(randn(n))

    #=
    Test hessian vector product
    =#
    @testset "hvp" begin
        v = randn(n)

        params, re = Flux.destructure(model)
        f(w, x) = re(w)(x)
        Hv = CubicNewton._hvp(f, params, data)

        @test Hv(v) ≈ 2*(data*data')*v
    end

    #=
    Test optimizer interface, just making sure everything runs
    =#
    # @testset "optimizer" begin
    #     opt = CubicNewtonOpt()
    #
    #     @test train!(model, data, opt) == true
    # end
end

#=
Test cubic newton optimizer
=#
@testset "optimizer" begin
    f(θ, x) = θ'*(x*x')*θ

    n = rand(5:10)
    params = randn(n)
    data = randn(n)

    opt = CubicNewtonOpt()

    @test opt(f, params, data) == true
end
