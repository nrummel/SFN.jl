#=
Author: Cooper Simpson

Wrapper around LinearOperator.jl based Hessian-vector product, no AD.
=#

using SFN: HvpOperator
using LinearOperators: LinearOperator

export LHvpOperator

#=

=#
mutable struct LHvpOperator{F, T<:AbstractFloat, S<:AbstractVector{T}, I<:Integer, L} <: HvpOperator
    f::F
    x::S
    op::L
    nProd::I
    power::I
end

#=
Base implementations for LHvpOperator
=#
Base.eltype(Hv::LHvpOperator{F, T, S, I, L}) where {F, T, S, I, L} = T
Base.size(Hv::LHvpOperator) = (size(Hv.x,1), size(Hv.x,1))
Base.adjoint(Hv::LHvpOperator) = Hv

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
function LHvpOperator(f::F, x::S; power::I=2) where {F, T<:AbstractFloat, S<:AbstractVector{T}, I<:Integer}
	return LHvpOperator(f, x, f(x), 0, power)
end

#=
Inplace matrix vector multiplcation with LHvpOperator.

Input:
	result :: matvec storage
	Hv :: LHvpOperator
	v :: rhs vector
=#
function apply!(result::S, Hv::LHvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
    Hv.nProd += 1

    mul!(result, Hv.op, v)

    return nothing
end