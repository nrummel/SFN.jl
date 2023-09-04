#=
Author: Cooper Simpson

Enzyme AD.
=#

using SFN: HvpOperator
using Enzyme: autodiff, autodiff_deferred, Forward, Reverse, Duplicated

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

    y = zero(T)
    dy = zero(T)
    by = one(T)
    dby = zero(T)

    autodiff(
        Forward,
        (x,y) -> autodiff_deferred(Reverse, f, x, y),
        Duplicated(Duplicated(x, bx), Duplicated(v, dbx)),
        Duplicated(Duplicated(y, by), Duplicated(dy, dby))
    )

    return dbx
end

#=
In-place hvp operator compatible with Krylov.jl
=#
mutable struct EHvpOperator{F, R<:AbstractFloat, S<:AbstractVector{R}} <: HvpOperator
    gf::F
    duplicated::Duplicated{S}
	nProd::Int
end

#=
Base implementations for EHvpOperator
=#
Base.eltype(Hv::EHvpOperator{F, R, S}) where {F, R, S} = R
Base.size(Hv::EHvpOperator) = (size(Hv.duplicated.val, 1), size(Hv.duplicated.val, 1))

#=
In place update of EHvpOperator
Input:
	x :: new input to f
=#
function update!(Hv::EHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
	Hv.duplicated = Duplicated(x, similar(x))
	Hv.nProd = 0

	return nothing
end

#=
Constructor.

Input:
	f :: scalar valued function
	x :: input to f
=#
function EHvpOperator(f::F, x::S) where {F, S<:AbstractVector{<:AbstractFloat}}
    gf = x -> autodiff_deferred(f, x)

	return EHvpOperator(gf, Duplicated(x, similar(x)), 0)
end

#=
Inplace matrix vector multiplcation with EHvpOperator.

Input:
	result :: matvec storage
	Hv :: EHvpOperator
	v :: rhs vector
=#
function apply!(result::S, Hv::EHvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	Hv.nProd += 1

    autodiff(
        Forward,
        Hv.gf,
        Duplicated(Hv.duplicated, Duplicated(v, result))
    )

	return nothing
end