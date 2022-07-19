#=
Author: Cooper Simpson

Associated functionality for matrix free Hessian vector multiplication operator
using ForwardDiff over ReverseDiff mixed mode AD.
=#

using ReverseDiff: AbstractTape, GradientTape, compile, gradient!, gradient
using ForwardDiff: Partials, partials, Dual, Tag

#=
Fast hessian vector product (hvp) function using ForwardDiff over ReverseDiff

Input:
	f :: scalar valued function
	x :: input to f
	v :: vector
=#
function _hvp(f::F, x::S, v::S) where {F, S<:AbstractVector{<:AbstractFloat}}
	dual = Dual.(x,v)

	return partials.(gradient(f, dual), 1)
end

#=
In-place hvp operator compatible with Krylov.jl
=#
mutable struct HvpOperator{F, R<:AbstractFloat, S<:AbstractVector{R}, T<:AbstractTape}
	x::S
	dualCache1::Vector{Dual{F, R, 1}}
	dualCache2::Vector{Dual{F, R, 1}}
	tape::T
	nProd::Int
end

#=
Base implementations for HvpOperator
=#
Base.eltype(Hop::HvpOperator{F, R, S, T}) where {F, R, S, T} = R
Base.size(Hop::HvpOperator) = (size(Hop.x,1), size(Hop.x,1))

#=
In place update of HvpOperator
Input:
	x :: new input to f
=#
function update!(Hop::HvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
	Hop.x .= x
	Hop.nProd = 0

	return nothing
end

#=
Constructor.

Input:
	f :: scalar valued function
	x :: input to f
=#
function HvpOperator(f::F, x::S, compile_tape=true) where {F, S<:AbstractVector{<:AbstractFloat}}
	dualCache1 = Dual{typeof(Tag(Nothing, eltype(x))),eltype(x),1}.(x, Partials.(Tuple.(similar(x))))
	dualCache2 = Dual{typeof(Tag(Nothing, eltype(x))),eltype(x),1}.(x, Partials.(Tuple.(similar(x))))

	tape = GradientTape(f, dualCache1)

	compile_tape ? tape = compile(tape) : tape

	return HvpOperator(x, dualCache1, dualCache2, tape, 0)
end

#=
Inplace matrix vector multiplcation with HvpOperator.

Input:
	result :: matvec storage
	Hop :: HvpOperator
	v :: rhs vector
=#
function apply!(result::S, Hop::HvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	Hop.nProd += 1

	Hop.dualCache1 .= Dual{typeof(Tag(Nothing, eltype(v))),eltype(v),1}.(Hop.x, Partials.(Tuple.(v)))

	gradient!(Hop.dualCache2, Hop.tape, Hop.dualCache1)

	result .= partials.(Hop.dualCache2, 1)

	return nothing
end

#=
Inplace matrix vector multiplcation with squared HvpOperator

Input:
	result :: matvec storage
	Hop :: HvpOperator
	v :: rhs vector
=#
function LinearAlgebra.mul!(result::S, Hop::HvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	apply!(result, Hop, v)
	apply!(result, Hop, result)

	return nothing
end
