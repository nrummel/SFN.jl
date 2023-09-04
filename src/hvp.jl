#=
Author: Cooper Simpson

Associated functionality for matrix free Hessian vector multiplication operator
using mixed mode AD.
=#

abstract type HvpOperator end

include("hvp/EnzymeHvpExt.jl")
include("hvp/LinopHvpExt.jl")
include("hvp/RdiffHvpExt.jl")
include("hvp/ZygoteHvpExt.jl")

#=
In-place matrix vector multiplcation with HvpOperator

WARNING: Default construction for Hv is power=2

Input:
	result :: matvec storage
	Hv :: HvpOperator
	v :: rhs vector
=#
function LinearAlgebra.mul!(result::S, Hv::HvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	apply!(result, Hv, v)

	for i=1:Hv.power-1
		apply!(result, Hv, result)
	end

	return nothing
end
