#=
Author: Cooper Simpson

Associated functionality for solving the cubic sub-problem in
cubic newton type methods.
=#

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
	op :: HvpOperator
	f :: scalar valued function
	x :: input to f
=#
# function update!(op::HvpOperator, f, x<:AbstractVector)
# 	op.f = f
# 	op.x = x
# 	return true
# end

#=
Base implementations for HvpOperator
=#
Base.eltype(op::HvpOperator{F, T, I}) where {F, T, I} = T
Base.size(op::HvpOperator) = (op.dim, op.dim)
Base.:*(op::HvpOperator, v::AbstractVector) = _hvp(op.f, op.x, v)

#=
Inplace matrix vector multiplcation with HvpOperator.

Input:
	result :: matvec storage
	op :: HvpOperator
	v :: rhs vector
=#
function LinearAlgebra.mul!(result::T, op::HvpOperator, v::T) where T<:AbstractVector
	op.nProd += 1

	op.dualCache1 .= Dual.(op.x, v)
	val, back = pullback(op.f, op.dualCache1)

	result .= partials.(back(one(val))[1], 1)
end

#=
Evaluate ∇f(x)ᵀd + 0.5dᵀ∇²f(x)d

Input:
	d :: descent direction
	grads :: gradients
	hess :: hessian operator
=#
function _quadratic_eval(d::T, grads::T, hess::HvpOperator, res::T) where T<:AbstractVector
	LinearAlgebra.mul!(res, hess, d)
	res .+= grads
	return dot(d, 0.5.*res)
end

#=
Evaluate ∇f(x)ᵀd + 0.5dᵀ∇²f(x)d + σ/3||d||³

Input:
	d :: descent direction
	grads :: gradients
	hess :: hessian operator
	σ :: cubic regularizer
=#
function _cubic_eval(d, grads, hess, σ, res)
	LinearAlgebra.mul!(res, hess, d)
	res .+= grads

    return dot(d, 0.5.*res) + (σ/3)*norm(d)^3
end
