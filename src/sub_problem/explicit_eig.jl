#=
Author: Cooper Simpson

Explicit eigenvector cubic sub-problem solver.
=#

#=
Solve the cubic sub-problem by minimizing the cubic upper bound to
obtain a descent direction.

Input:
	f -> target function
	params -> parameters of f
	subsample -> subset of full inputs
	grads -> gradients of f w.r.t params
	σ -> cubic regularizer
=#
function _cubic_minimize(f, params, subsample, grads, σ)
	return grads, 1
end
