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

	return partials.(back(one.(val))[1], 1)
end

#=
In-place hvp operator compatible with Krylov.jl
=#
mutable struct HvpOperator{F, T, I}
	f::F
	x::AbstractArray{T, 1}
	dualCache1::Vector{Dual{Nothing, T, 1}}
	size::I
	nProd::I
end

function HvpOperator(f, x::AbstractVector)
	dualCache1 = Dual.(x, similar(x))
	return HvpOperator(f, x, dualCache1, size(x, 1), 0)
end

Base.eltype(op::HvpOperator{F, T, I}) where{F, T, I} = T
Base.size(op::HvpOperator) = (op.size, op.size)
Base.:*(op::HvpOperator, v::AbstractVector) = _hvp(op.f, op.x, v)

function LinearAlgebra.mul!(result::AbstractVector, op::HvpOperator, v::AbstractVector)
	op.nProd += 1

	op.dualCache1 .= Dual.(op.x, v)
	val, back = pullback(op.f, op.dualCache1)

	result .= partials.(back(one.(val))[1], 1)
end

#=
Evaluate ∇f(x)ᵀd + 0.5dᵀ∇²f(x)d

Input:
	d :: descent direction
	grads :: gradients
	hess :: hessian operator
=#
function _quadratic_eval(d, grads, hess)
	return dot(d, grads) + 0.5*dot(d, hess*d)
end

#=
Evaluate ∇f(x)ᵀd + 0.5dᵀ∇²f(x)d + σ/3||d||³

Input:
	d :: descent direction
	grads :: gradients
	hess :: hessian operator
	σ :: cubic regularizer
=#
function _cubic_eval(d, grads, hess, σ)
    return dot(d, grads) + 0.5*dot(d, hess*d) + (σ/3)*norm(d)^3
end
