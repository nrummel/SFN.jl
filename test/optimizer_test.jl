#=
Author: Cooper Simpson

Tests for functionality found in src/optimizer.jl -- the RSFN optimizer.
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
        Test ShiftedLanczosCG optimizer
        =#
        @testset "shifted lanczos cg" begin
            opt = ShiftedLanczosCG()
            x = [0.0, 3.0]

            @test_nowarn minimize!(opt, x, rosenbrock, itmax=5)
        end
    end
end
