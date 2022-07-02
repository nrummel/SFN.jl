using LinearAlgebra
using Krylov: CgLanczosSolver, cg_lanczos!
using Zygote: pullback
using ForwardDiff: partials, Dual

function rosenbrock(x::T) where T<:AbstractVector
    res = 0.0
    for i = 1:size(x,1)-1
        res += 100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
    end
    return res
end

mutable struct HvpOperator{F, T<:AbstractFloat, S<:AbstractVector{T}}
	f::F
	dim::Int
	x::S
	dualCache1::AbstractVector{Dual{Nothing, T, 1}}
	nProd::Int
end

function Hvp(f::F, x::S) where {F, S<:AbstractVector{<:AbstractFloat}}
	dualCache1 = Dual.(x, similar(x))

	return HvpOperator(f, size(x, 1), x, dualCache1, 0)
end

Base.eltype(Hop::HvpOperator{F, T, S}) where {F, T, S} = T
Base.size(Hop::HvpOperator) = (Hop.dim, Hop.dim)

function LinearAlgebra.mul!(result::S, Hop::HvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	Hop.nProd += 1

	Hop.dualCache1 .= Dual.(Hop.x, v)
	val, back = pullback(Hop.f, Hop.dualCache1)

	result .= partials.(back(one(val))[1], 1)

	return nothing
end

function newton!(x::S, f::F; itmax::Int=1000) where {S<:AbstractVector{<:AbstractFloat}, F}
    solver = CgLanczosSolver(size(x,1), size(x,1), S)

    grads = similar(x)

    for i = 1:itmax
        #construct gradient and hvp operator
        loss, back = pullback(f, x)
        grads .= back(one(loss))[1]

        Hop = Hvp(f, x)

        cg_lanczos!(solver, Hop, grads)

        x .-= solver.x
    end

    return nothing
end
