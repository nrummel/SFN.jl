
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
@add_arg_table parse begin
    "--function, -f"
        arg_type = String
    "--dimension, -d"
        arg_type = Int
        default = 10
    "--itmax"
        arg_type = Int
        default = 1000
end

args = parse_args(ARGS, parser)

func = eval(Meta.parse(args["function"]))
dim = args["dimension"]

if func == "michalewicz":
    func = michalewicz
    x = randn(dim)
elseif func == "rosenbrock"
    func = rosenbrock
    x = randn(dim)
elseif func == "matfact"
    func = matfact
    x = randn(dim)

#Optimize
opt = RSFNOptimizer(size(x, 1))

minimize!(opt, x, rosenbrock, itmax=args["itmax"])

exact = ones(dim)

println(LA.norm(exact-x)/LA.norm(exact))
