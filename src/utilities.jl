#=
Author: Cooper Simpson

Associated functionality for solving the cubic sub-problem in
cubic newton type methods.
=#

#=
Builds a fast hessian vector product (hvp) function using forward-over-back AD

Input:
	f :: scalar valued function
	x :: input to f
=#
function _hvop(f, x)
	g = x -> gradient(f, x)[1]
	H(v) = partials.(g(Dual.(params, v)), 1)

	return H
end

#=
Builds in-place hvp function

Input:
	results :: result of hvp
	dual_cache1 :: cache for Dual.(inputs,v)
	dual_cache2 :: cache for Dual.(inputs,v)
	f :: scalar valued function
	x :: input to f
=#
function _hvop!(results, dual_cache1, dual_cache2, f, x)
	g = (dx, x) -> dx .= gradient(f, x)[1]

	H(v) = begin
		dual_cache1 .= Dual.(params, v)
		g(dual_cache2, dual_cache1)
		results .= partials.(dual_cache2, 1)
	end

	return H
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
