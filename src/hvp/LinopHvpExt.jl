#=
Author: Cooper Simpson

Wrapper around LinearOperator.jl based Hessian-vector product, no AD.
=#

using SFN: HvpOperator
using LinearOperators: LinearOperator

export LHvpOperator

#=

=#
mutable struct LHvpOperator{R<:AbstractFloat, S<:AbstractVector{R}, F} <: HvpOperator
    x::S
    build::F
    hessian::LinearOperator
    nProd::Int
end

#=
Base implementations for LHvpOperator
=#
Base.eltype(Hv::LHvpOperator{R, S, F}) where {R, S, F} = R
Base.size(Hv::LHvpOperator) = (size(Hv.x,1), size(Hv.x,1))

#=
In place update of LHvpOperator
Input:
	x :: new input to f
=#
function update!(Hv::LHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
	Hv.x .= x
    Hv.hessian = Hv.build(x)
	Hv.nProd = 0

	return nothing
end

#=
Constructor.

Input:
	x :: input to f
    build :: function that builds hessian operator
=#
function LHvpOperator(x::S, build::F) where {S<:AbstractVector{<:AbstractFloat}, F}
	return LHvpOperator(x, build, build(x), 0)
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

    mul!(result, Hv.H, v)

    return nothing
end