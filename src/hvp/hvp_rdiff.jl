#=
Author: Cooper Simpson

ForwardDiff over ReverseDiff AD.
=#

using ReverseDiff: AbstractTape, GradientTape, compile, gradient!, gradient
using ForwardDiff: Partials, partials, Dual, Tag

#=
Fast hessian vector product (hvp) function.

Input:
	f :: scalar valued function
	x :: input to f
	v :: vector
=#
function rhvp(f::F, x::S, v::S) where {F, S<:AbstractVector{<:AbstractFloat}}
	dual = Dual.(x,v)

	return partials.(gradient(f, dual), 1)
end

#=
In-place hvp operator compatible with Krylov.jl
=#
mutable struct RHvpOperator{F, R<:AbstractFloat, S<:AbstractVector{R}, T<:AbstractTape} <: HvpOperator
	x::S
	dualCache1::Vector{Dual{F, R, 1}}
	dualCache2::Vector{Dual{F, R, 1}}
	tape::T
	nProd::Int
end

#=
Base implementations for RHvpOperator
=#
Base.eltype(Hv::RHvpOperator{F, R, S, T}) where {F, R, S, T} = R
Base.size(Hv::RHvpOperator) = (size(Hv.x,1), size(Hv.x,1))

#=
In place update of RHvpOperator
Input:
	x :: new input to f
=#
function update!(Hv::RHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
	Hv.x .= x
	Hv.nProd = 0

	return nothing
end

#=
Constructor.

Input:
	f :: scalar valued function
	x :: input to f
=#
function RHvpOperator(f::F, x::S, compile_tape=true) where {F, S<:AbstractVector{<:AbstractFloat}}
	dualCache1 = Dual{typeof(Tag(Nothing, eltype(x))),eltype(x),1}.(x, Partials.(Tuple.(similar(x))))
	dualCache2 = Dual{typeof(Tag(Nothing, eltype(x))),eltype(x),1}.(x, Partials.(Tuple.(similar(x))))

	tape = GradientTape(f, dualCache1)

	compile_tape ? tape = compile(tape) : tape

	return RHvpOperator(x, dualCache1, dualCache2, tape, 0)
end

#=
Inplace matrix vector multiplcation with RHvpOperator.

Input:
	result :: matvec storage
	Hv :: RHvpOperator
	v :: rhs vector
=#
function apply!(result::S, Hv::RHvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	Hv.nProd += 1

	Hv.dualCache1 .= Dual{typeof(Tag(Nothing, eltype(v))),eltype(v),1}.(Hv.x, Partials.(Tuple.(v)))

	gradient!(Hv.dualCache2, Hv.tape, Hv.dualCache1)

	result .= partials.(Hv.dualCache2, 1)

	return nothing
end
