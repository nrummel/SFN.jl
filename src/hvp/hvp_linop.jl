#=
Author: Cooper Simpson

Wrapper around LinearOperator.jl based Hessian-vector product, no AD.
=#

using LinearOperators: LinearOperator

mutable struct LHvpOperator{R<:AbstractFloat, S<:AbstractVector{R}, F} <: HvpOperator
    x::S
    h_build::F
    hop::LinearOperator
    nProd::Int
end

#=
Base implementations for LHvpOperator
=#
Base.eltype(Hop::LHvpOperator{R, S, F}) where {R, S, F} = R
Base.size(Hop::LHvpOperator) = (size(Hop.x,1), size(Hop.x,1))

#=
In place update of LHvpOperator
Input:
	x :: new input to f
=#
function update!(Hop::LHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
	Hop.x .= x
    Hop.hop = Hop.h_build(x)
	Hop.nProd = 0

	return nothing
end

#=
Constructor.

Input:
	x :: input to f
    h_build :: function that builds hessian operator
=#
function LHvpOperator(x::S, h_build::F) where {S<:AbstractVector{<:AbstractFloat}, F}
	return LHvpOperator(x, h_build, h_build(x), 0)
end

#=
Inplace matrix vector multiplcation with LHvpOperator.

Input:
	result :: matvec storage
	Hop :: LHvpOperator
	v :: rhs vector
=#
function apply!(result::S, Hop::LHvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
    Hop.nProd += 1

    mul!(result, Hop.hop, v)

    return nothing
end
