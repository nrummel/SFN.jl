import LinearAlgebra as LA
using Krylov: CgLanczosShiftSolver, cg_lanczos!
using Zygote: pullback, gradient
using ForwardDiff: partials, Dual, hessian
using RSFN: Logger, RSFNOptimizer, RHvpOperator, step!
using BenchmarkTools

function rosenbrock(x::T) where T<:AbstractVector
    res = 0.0
    for i = 1:size(x,1)-1
        res += 100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
    end
    return res
end

function gradient_descent!(x::S, f::F; itmax::Int=1000) where {S<:AbstractVector{<:AbstractFloat}, F}
    grads = similar(x)

    for i = 1:itmax
        #construct gradient and hvp operator
        loss, back = pullback(f, x)
        grads .= back(one(loss))[1]

        x .-= LA.dot(grads, x)
    end

    return nothing
end

mutable struct HvpOp{F, T<:AbstractFloat, S<:AbstractVector{T}}
	f::F
	dim::Int
	x::S
	dualCache1::AbstractVector{Dual{Nothing, T, 1}}
	nProd::Int
end

function HvpOp(f::F, x::S) where {F, S<:AbstractVector{<:AbstractFloat}}
	dualCache1 = Dual.(x, similar(x))

	return HvpOp(f, size(x, 1), x, dualCache1, 0)
end

Base.eltype(Hop::HvpOp{F, T, S}) where {F, T, S} = T
Base.size(Hop::HvpOp) = (Hop.dim, Hop.dim)

function LA.mul!(result::S, Hop::HvpOp, v::S) where S<:AbstractVector{<:AbstractFloat}
	Hop.nProd += 1

	Hop.dualCache1 .= Dual.(Hop.x, v)
	val, back = pullback(Hop.f, Hop.dualCache1)

	result .= partials.(back(one(val))[1], 1)

	return nothing
end

function newton!(x::S, f::F; itmax::Int=1000) where {S<:AbstractVector{<:AbstractFloat}, F}
    solver = CgLanczosShiftSolver(size(x,1), size(x,1), 1, S)
    logger = Logger()

    grads = similar(x)

    for i = 1:itmax
        #construct gradient and hvp operator
        loss, back = pullback(f, x)
        grads .= back(one(loss))[1]
        logger.gcalls += 1

        Hop = HvpOp(f, x)

        newton_step!(solver, x, f, grads, Hop)

        logger.hcalls += Hop.nProd
    end

    return logger
end

function newton_step!(solver::CgLanczosShiftSolver, x::S, f::F, grads::S, Hop::HvpOp) where {S<:AbstractVector{<:AbstractFloat}, F}
    cg_lanczos!(solver, Hop, grads, 1e-6)

    x .-= solver.x[1]
end

function newton_step!(x::S, f::F, grads::S, hess::H) where {S<:AbstractVector{<:AbstractFloat}, F, H}
    x .-= LA.Symmetric(hess)\grads
end

function sfn_step!(x::S, f::F, grads::S, hess::H) where {S<:AbstractVector{<:AbstractFloat}, F, H}
    D, V = LA.eigen(LA.Symmetric(hess))
    @. D = inv(abs(D))
    hess .= V*LA.Diagonal(D)*transpose(V)

    x .-= hess*grads
end

function eval_newton(f, n)
	solver = CgLanczosShiftSolver(n, n, 1, Vector{Float64})

	return @benchmark newton_step!($solver, x, $f, grads, HvpOp($f, x)) setup=begin; x=randn($n); grads=gradient($f,x)[1];end
end

function eval_newton(f, H, n)
	return @benchmark newton_step!(x, $f, grads, $H(x)) setup=begin; x=randn($n); grads=gradient($f,x)[1];end
end

function eval_sfn(f, n)
	return @benchmark sfn_step!(x, $f, grads, hessian($f, x)) setup=begin; x=randn($n); grads=gradient($f,x)[1];end
end

function eval_sfn(f, H, n)
	return @benchmark sfn_step!(x, $f, grads, $H(x)) setup=begin; x=randn($n); grads=gradient($f,x)[1];end
end

function eval_rsfn(f, n, quad_order=20)
	opt = RSFNOptimizer(n, M=0, ϵ=0.0, quad_order=quad_order)

	return @benchmark step!($opt, x, $f, grads, RHvpOperator($f, x)) setup=begin; x=randn($n); grads=gradient($f,x)[1];end
end
