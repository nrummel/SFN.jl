#=
Author: Cooper Simpson

Tests for functionality found in src/optimizer.jl -- the SFN optimizer.
=#

if run_all || "optimizer" in ARGS
    @testset "optimizer" begin

        function rosenbrock(x)
            res = 0.0
            for i = 1:size(x,1)-1
                res += 100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
            end
            return res
        end

        #=
        Test SFN optimizer
        =#
        @testset "SFN optimizer" begin
            x = [0.0, 3.0]

            opt = SFNOptimizer(size(x,1))

            @test_nowarn minimize!(opt, x, rosenbrock, itmax=5)
        end
    end
end
