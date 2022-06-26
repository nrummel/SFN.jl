
#=
Author: Cooper Simpson

Convex experiment.
=#

using RSFN

function f()::Float64
    return 1.0
end

x = [1.0, 2.0]

opt = RSFNOptimizer(size(x, 1))

minimize!(opt, x, f)
