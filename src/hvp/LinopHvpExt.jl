#=
Author: Cooper Simpson

Wrapper around LinearOperator.jl based Hessian-vector product, no AD.
=#

using SFN: HvpOperator
using LinearOperators: LinearOperator

export LHvpOperator

#=

=#
mutable struct LHvpOperator{F, T<:AbstractFloat, S<:AbstractVector{T}, I<:Integer, L} <: HvpOperator{T}
    f::F
    x::S
    op::L
    nprod::I
    power::I
end
# extend Matrix for the trivial case when the operator is a matrix
function Base.Matrix(Hvp::LHvpOperator{F,T,S,I,L}) where {F,T,S,I,L<:AbstractMatrix}
    return Hvp.op
end

#=
In place update of LHvpOperator
Input:
	x :: new input to f
=#
function update!(Hv::LHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
	Hv.x .= x
    Hv.op = Hv.f(x)

	return nothing
end

#=
Constructor.

Input:
    f :: function that builds hessian operator
	x :: input to f
=#
function LHvpOperator(f::F, x::S; power::I=1) where {F, T<:AbstractFloat, S<:AbstractVector{T}, I<:Integer}
	return LHvpOperator(f, x, f(x), 0, power)
end

#=
Inplace matrix vector multiplcation with LHvpOperator.

Input:
	result :: matvec storage
	Hv :: LHvpOperator
	v :: rhs vector
=#
function apply!(result::AbstractVector, Hv::LHvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
    Hv.nprod += 1

    mul!(result, Hv.op, v)

    return nothing
end