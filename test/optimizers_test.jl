#=
Author: Cooper Simpson

Tests for functionality found in src/optimizers.jl -- the specific CubicNewton
optimizers.
=#

if run_all || "optimizers" in ARGS
    @testset "optimizers" begin

        #=
        Test ShiftedLanczosCG optimizer
        =#
        @testset "shifted lanczos cg" begin
            #TODO
        end

        #=
        Test Eigen optimizer
        =#
        @testset "eigen" begin
            #TODO
        end
    end
end
