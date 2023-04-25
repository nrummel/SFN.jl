#=
Author: Cooper Simpson

Associated functionality for matrix free Hessian vector multiplication operator
using mixed mode AD.
=#

abstract type HvpOperator end

#=
Inplace matrix vector multiplcation with squared HvpOperator

Input:
	result :: matvec storage
	Hv :: HvpOperator
	v :: rhs vector
=#
function LinearAlgebra.mul!(result::S, Hv::HvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	apply!(result, Hv, v)
	apply!(result, Hv, result)

	return nothing
end
