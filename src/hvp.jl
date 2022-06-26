#=
Author: Cooper Simpson

Associated functionality for matrix free Hessian vector multiplication operator.
=#

using Zygote: pullback
using ForwardDiff: partials, Dual

#=
Fast hessian vector product (hvp) function using forward-over-back AD

Input:
	f :: scalar valued function
	x :: input to f
	v :: vector
=#
function _hvp(f, x, v)
	val, back = pullback(f, Dual.(x,v))
	return partials.(back(one(val))[1], 1)
end

#=
In-place hvp operator compatible with Krylov.jl
=#
mutable struct HvpOperator{F, T, I<:Int}
	f::F
	x::AbstractVector{T}
	dualCache1::AbstractArray{Dual{Nothing, T, 1}}
	dim::I
	nProd::I
end

#=
Constructor.

Input:
	type :: data type of operator
	size :: size of operator
=#
# function HvpOperator(type<:AbstractVector, size<:Int)
# 	x = type(undef, size)
# 	dualCache1 = Dual.(x, similar(x))
# 	return HvpOperator(() -> (), x, dualCache1, size, 0)
# end

#=
Constructor.

Input:
	f :: scalar valued function
	x :: input to f
=#
function HvpOperator(f::Function, x::T) where T<:AbstractVector
	dualCache1 = Dual.(x, similar(x))
	return HvpOperator(f, x, dualCache1, size(x, 1), 0)
end

#=
Update internals of operator inplace.

Input:
	Hop :: HvpOperator
	f :: scalar valued function
	x :: input to f
=#
# function update!(Hop::HvpOperator, f, x<:AbstractVector)
# 	Hop.f = f
# 	Hop.x = x
# 	return true
# end

#=
Base implementations for HvpOperator
=#
Base.eltype(Hop::HvpOperator{F, T, I}) where {F, T, I} = T
Base.size(Hop::HvpOperator) = (Hop.dim, Hop.dim)
Base.:*(Hop::HvpOperator, v::AbstractVector) = _hvp(Hop.f, Hop.x, v)

#=
Inplace matrix vector multiplcation with HvpOperator.

Input:
	result :: matvec storage
	Hop :: HvpOperator
	v :: rhs vector
=#
function apply!(result::T, Hop::HvpOperator, v::T) where T<:AbstractVector
	Hop.nProd += 1

	Hop.dualCache1 .= Dual.(Hop.x, v)
	val, back = pullback(Hop.f, Hop.dualCache1)

	result .= partials.(back(one(val))[1], 1)
end

#=
Inplace matrix vector multiplcation with squared HvpOperator

Input:
	result :: matvec storage
	Hop :: HvpOperator
	v :: rhs vector
=#
function LinearAlgebra.mul!(result::T, Hop::HvpOperator, v::T) where T<:AbstractVector
	apply!(result, Hop, v)
	apply!(result, Hop, result)
end
