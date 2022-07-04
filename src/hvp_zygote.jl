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
function _hvp_zygote(f::F, x::S, v::S) where {F, S<:AbstractVector{<:AbstractFloat}}
	val, back = pullback(f, Dual.(x,v))

	return partials.(back(one(val))[1], 1)
end

#=
In-place hvp operator compatible with Krylov.jl
=#
mutable struct ZHvpOperator{F, T<:AbstractFloat, S<:AbstractVector{T}}
	f::F
	x::S
	dualCache1::AbstractVector{Dual{Nothing,T,1}}
	nProd::Int
end

#=
Base implementations for HvpOperator
=#
Base.eltype(Hop::ZHvpOperator{F, T, S}) where {F, T, S} = T
Base.size(Hop::ZHvpOperator) = (size(Hop.x,1), size(Hop.x,1))

#=
Constructor.

Input:
	f :: scalar valued function
	x :: input to f
=#
function ZHvpOperator(f::F, x::S) where {F, S<:AbstractVector{<:AbstractFloat}}
	dualCache1 = Dual.(x, similar(x))

	return ZHvpOperator(f, x, dualCache1, 0)
end

#=
Inplace matrix vector multiplcation with HvpOperator.

Input:
	result :: matvec storage
	Hop :: HvpOperator
	v :: rhs vector
=#
function apply!(result::S, Hop::ZHvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	Hop.nProd += 1

	Hop.dualCache1 .= Dual.(Hop.x, v)
	val, back = pullback(Hop.f, Hop.dualCache1)

	result .= partials.(back(one(val))[1], 1)

	return nothing
end

#=
Inplace matrix vector multiplcation with squared HvpOperator

Input:
	result :: matvec storage
	Hop :: HvpOperator
	v :: rhs vector
=#
function LinearAlgebra.mul!(result::S, Hop::ZHvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	apply!(result, Hop, v)
	apply!(result, Hop, result)

	return nothing
end
