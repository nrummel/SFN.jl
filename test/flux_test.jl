#=
Author: Cooper Simpson

Tests for functionality found in src/flux.jl -- the CubicNewton interface with
the Flux.jl package.
=#

using Flux

if run_all || "flux" in ARGS
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
        Test basic hessian vector product
        =#
        @testset "hvp" begin
            v = randn(n)

            params, re = Flux.destructure(model)
            f(w, x) = re(w)(x)
            Hv = CubicNewton._hvp(f, params, data)

            @test Hv(v) ≈ 2*(data*data')*v

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
