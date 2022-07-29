import LinearAlgebra as LA
using Krylov: CgLanczosShiftSolver, cg_lanczos_shift!
using Zygote: gradient
using ReverseDiff: AbstractTape, GradientTape, compile, gradient!
using ForwardDiff: Partials, partials, Dual, hessian, Tag
using RSFN: Logger, RSFNOptimizer, RHvpOperator, step!
using BenchmarkTools
using NNlib: logsigmoid, sigmoid
using MLJ: make_regression
using Statistics

#=
Test problems
=#

function rosenbrock(x::T) where T<:AbstractVector
    res = 0.0
    for i = 1:size(x,1)-1
        res += 100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
    end
    return res
end

function logistic_loss(x, y)
	return -sum(logsigmoid.(x.*y))
end

function accuracy(θ, X, y)
	return acc = sum(map(x -> x<0 ? -1 : 1, X*θ) .== y)/size(X,1)
end

function make_data(num_feat)
	num_obs = 1000
	X, y = make_regression(num_obs, num_feat; rng=42, as_table=false, binary=false, intercept=false, noise=1)
	y = map(x -> x<0 ? -1 : 1, y)

	f(w) = logistic_loss(X*w, y)

	return X, y, f
end

function log_hess(w, X, y)
	u = sigmoid.(-y.*(X*w))
	S = LA.Diagonal(u.*(ones(size(u,1))-u))
	return transpose(X)*S*X
end

#=
Optimization Methods
=#

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

mutable struct HvpOp{F, R<:AbstractFloat, S<:AbstractVector{R}, T<:AbstractTape}
	x::S
	dualCache1::Vector{Dual{F, R, 1}}
	dualCache2::Vector{Dual{F, R, 1}}
	tape::T
	nProd::Int
end

function HvpOp(f::F, x::S, compile_tape=true) where {F, S<:AbstractVector{<:AbstractFloat}}
	dualCache1 = Dual{typeof(Tag(Nothing, eltype(x))),eltype(x),1}.(x, Partials.(Tuple.(similar(x))))
	dualCache2 = Dual{typeof(Tag(Nothing, eltype(x))),eltype(x),1}.(x, Partials.(Tuple.(similar(x))))

	tape = GradientTape(f, dualCache1)

	compile_tape ? tape = compile(tape) : tape

	return HvpOp(x, dualCache1, dualCache2, tape, 0)
end

Base.eltype(Hop::HvpOp{F, R, S, T}) where {F, R, S, T} = R
Base.size(Hop::HvpOp) = (size(Hop.x,1), size(Hop.x,1))

function LA.mul!(result::S, Hop::HvpOp, v::S) where S<:AbstractVector{<:AbstractFloat}
	Hop.nProd += 1

	Hop.dualCache1 .= Dual{typeof(Tag(Nothing, eltype(v))),eltype(v),1}.(Hop.x, Partials.(Tuple.(v)))

	gradient!(Hop.dualCache2, Hop.tape, Hop.dualCache1)

	result .= partials.(Hop.dualCache2, 1)

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
    cg_lanczos_shift!(solver, Hop, grads, [1e-6])

    x .-= solver.x[1]
end

function newton_step!(x::S, f::F, grads::S, hess::H) where {S<:AbstractVector{<:AbstractFloat}, F, H}
    x .-= LA.Symmetric(hess+1e-6*LA.I(size(hess,2)))\grads
end

function sfn_step!(x::S, f::F, grads::S, hess::H) where {S<:AbstractVector{<:AbstractFloat}, F, H}
    D, V = LA.eigen(LA.Symmetric(hess + 1e-6*LA.I(size(hess,2))))
    @. D = inv(abs(D))
    hess .= V*LA.Diagonal(D)*transpose(V)

    x .-= hess*grads
end

#=
Benchmark Functions
=#

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
	opt = RSFNOptimizer(n, M=0, ϵ=1e-6, quad_order=quad_order)

	return @benchmark step!($opt, x, $f, grads, RHvpOperator($f, x)) setup=begin; x=randn($n); grads=gradient($f,x)[1];end
end
