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
	Hop :: HvpOperator
	v :: rhs vector
=#
function LinearAlgebra.mul!(result::S, Hop::HvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	apply!(result, Hop, v)
	apply!(result, Hop, result)

	return nothing
end
