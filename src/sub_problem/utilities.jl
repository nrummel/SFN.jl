#=
Author: Cooper Simpson

Associated functionality for solving the cubic sub-problem in
cubic newton type methods.
=#

#=
Builds a fast hessian vector product (hvp) function using forward-over-back AD

Input:
	f -> function
	params -> parameters of f
	inputs -> inputs to f
=#
function _hvp(f, params, inputs)
	g = θ -> gradient(θ -> f(inputs, θ), θ)[1]
	Hv(v) = partials.(g(Dual.(params, v)), 1)

	return Hv
end

#=
Builds in-place hvp function

Input:
	results -> result of hvp
	dual_cache1 -> cache for Dual.(inputs,v)
	dual_cache2 -> cache for Dual.(inputs,v)
	f -> function
	params -> parameters of f
	inputs -> inputs to f
=#
function _hvp!(results, dual_cache1, dual_cache2, f, params, inputs)
	g = (dθ, θ) -> dθ .= gradient(θ -> f(inputs, θ), θ)[1]

	Hv(v) = begin
		dual_cache1 .= Dual.(params, v)
		g(dual_cache2, dual_cache1)
		results .= partials.(dual_cache2, 1)
	end

	return Hv
end

#=
Evaluate the cubic upper bound and its gradient

Input:
	hvp -> hvp function
	s -> descent direction
	grads -> gradients
	σ -> cubic regularizer
=#
function _cubic_eval(hvp, s, grads, σ)
    Hs = hvp(s)
    s_norm = norm(s)

    cubic_val = dot(s, grads) + 0.5*dot(s, Hs) + (σ/3)*s_norm^3
    cubic_grad = g + Hs + σ*s_norm*s

    return cubic_val, cubic_grad
end
