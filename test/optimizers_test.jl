#=
Author: Cooper Simpson

Tests for functionality found in src/optimizers.jl -- the specific CubicNewton
optimizers.
=#

if run_all || "optimizers" in ARGS
    @testset "optimizers" begin

        function rosenbrock(x)
            res = 0.0
            for i = 1:size(x,1)-1
                res += 100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
            end
            return res
        end

        #=
        Test ShiftedLanczosCG optimizer
        =#
        @testset "shifted lanczos cg" begin
            opt = ShiftedLanczosCG()
            x = [0.0, 3.0]

            @test_nowarn minimize!(opt, rosenbrock, x, itmax=5) #just make sure it runs
        end

        #=
        Test Eigen optimizer
        =#
        # @testset "eigen" begin
        #     #TODO
        # end
    end
end
