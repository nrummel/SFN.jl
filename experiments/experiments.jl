
#=
Author: Cooper Simpson

Non-convex experiment.
=#

using Pkg

Pkg.activate(".")

using RSFN
import LinearAlgebra as LA
using Random
using ArgParse

include("functions.jl")

#Parse arguments
parser = ArgParseSettings()
@add_arg_table parser begin
    "--function", "-f"
        arg_type = String
    "--dimension", "-d"
        arg_type = Int
        default = 10
    "--itmax"
        arg_type = Int
        default = 1000
end

args = parse_args(ARGS, parser)

dim = args["dimension"]

if args["function"] == "rosenbrock"
    func = rosenbrock
    x = zeros(dim)
    exact = 0.0
elseif args["function"] == "matfact"
    M = randn((dim, dim*2)) #make this rank r
    r = 2
    func = x -> matfact(x, M, r)
    x = randn(r*sum(size(M)))
    exact = 0.0
end

#Optimize
opt = RSFNOptimizer(size(x, 1), M=10, quad_order=100)

@time minimize!(opt, x, rosenbrock, itmax=args["itmax"])
println(abs(func(x)-exact))
