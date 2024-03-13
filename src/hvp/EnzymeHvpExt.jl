#=
Author: Cooper Simpson

Enzyme AD.
=#

using SFN: HvpOperator
using Enzyme: autodiff, autodiff_deferred, Forward, Reverse, Active, Const, Duplicated, DuplicatedNoNeed

export ehvp, EHvpOperator

#=
Fast hessian vector product (hvp) function.

Input:
	f :: scalar valued function
	x :: input to f
	v :: vector
=#
function ehvp(f::F, x::S, v::S) where {F, T<:AbstractFloat, S<:AbstractVector{T}}

    bx = similar(x)
    dbx = similar(x)

    autodiff(
        Forward,
        x -> autodiff_deferred(Reverse, f, Active, x),
        Const,
        DuplicatedNoNeed(Duplicated(x, bx), Duplicated(v, dbx))
    )

    return dbx
end

#=
In-place hvp operator compatible with Krylov.jl
=#
mutable struct EHvpOperator{F, T<:AbstractFloat, S<:AbstractVector{T}} <: HvpOperator{T}
    gf::F
    duplicated::Duplicated{S}
	nprod::Integer
    power::Integer
end

#=
Base implementations for EHvpOperator
=#
function Base.size(Hv::EHvpOperator)
    n = size(Hv.duplicated.val, 1)
    
    return (n,n)
end

#=
In place update of EHvpOperator
Input:
	x :: new input to f
=#
function update!(Hv::EHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
    # Hv.duplicated.val .= x
    Hv.duplicated = Duplicated(x, similar(x))

	return nothing
end

#=
Constructor.

Input:
	f :: scalar valued function
	x :: input to f
=#
function EHvpOperator(f::F, x::S; power::Integer=1) where {F, T<:AbstractFloat, S<:AbstractVector{T}}
    gf = x -> autodiff_deferred(Reverse, f, Active, x)

	return EHvpOperator(gf, Duplicated(x, similar(x)), 0, power)
end

#=
Inplace matrix vector multiplcation with EHvpOperator.

Input:
	result :: matvec storage
	Hv :: EHvpOperator
	v :: rhs vector
=#
function apply!(result::S, Hv::EHvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	Hv.nprod += 1

    autodiff(
        Forward,
        Hv.gf,
        Const,
        DuplicatedNoNeed(Hv.duplicated, Duplicated(v, result))
    )

	return nothing
end