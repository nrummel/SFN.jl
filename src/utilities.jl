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
	g = x -> gradient(f, x)[1]
	return partials.(g(Dual.(params, v)), 1)
end

#=
Builds in-place hvp function

Input:
	result :: result of hvp
	dual_cache1 :: cache for Dual.(inputs,v)
	dual_cache2 :: cache for Dual.(inputs,v)
	f :: scalar valued function
	x :: input to f
=#

mutable struct HVPOperator{F, T<:Number, I<:Integer}
	f::F #scalar valued function
	dual::Dual{T, T} #cache for Dual.(x,v)
	size::I #size of operator
	nprod::UInt16 #number of applications
end
HVPOperator(f, x) = HVPOperator(f, Dual.(x, similar(x)), size(x))

Base.eltype(op::HVPOperator{F, T, I}) where{F, T, I} = T
Base.size(op::HVPOperator) = (op.size, op.size)

function mul!(result::AbstractVector, op::HVPOperator, v::AbstractVector)
	op.dual_cache1.values .= v
	result .= partials.(gradient(op.f, op.dual)[1], 1)
end

#=
Evaluate ∇f(x)ᵀd + 0.5dᵀ∇²f(x)d

Input:
	d :: descent direction
	grads :: gradients
	hess :: hessian operator
=#
function _quadratic_eval(s, grads, hess)
	return dot(d, grads) + 0.5*dot(d, hess(d))
end

#=
Evaluate ∇f(x)ᵀd + 0.5dᵀ∇²f(x)d + σ/3||d||³

Input:
	d :: descent direction
	grads :: gradients
	hess :: hessian operator
	σ :: cubic regularizer
=#
function _cubic_eval(s, grads, hess, σ)
    return dot(d, grads) + 0.5*dot(d, hess(d)) + (σ/3)*norm(d)^3
end
