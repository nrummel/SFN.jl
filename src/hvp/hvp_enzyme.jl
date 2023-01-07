#=
Author: Cooper Simpson

Enzyme AD.
=#

using Enzyme: autodiff, autodiff_deferred, Forward, Duplicated

#=
Fast hessian vector product (hvp) function.

Input:
	f :: scalar valued function
	x :: input to f
	v :: vector
=#
function ehvp(f::F, x::S, v::S) where {F, S<:AbstractVector{<:AbstractFloat}}

    bx = similar(x)
    dbx = similar(x)

    autodiff(
        Forward,
        x -> autodiff_deferred(f, x),
        Duplicated(Duplicated(x, bx), Duplicated(v, dbx))
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
Base.eltype(Hop::EHvpOperator{F, R, S}) where {F, R, S} = R
Base.size(Hop::EHvpOperator) = (size(Hop.duplicated.val, 1), size(Hop.duplicated.val, 1))

#=
In place update of EHvpOperator
Input:
	x :: new input to f
=#
function update!(Hop::EHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
	Hop.duplicated = Duplicated(x, similar(x))
	Hop.nProd = 0

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
	Hop :: EHvpOperator
	v :: rhs vector
=#
function apply!(result::S, Hop::EHvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	Hop.nProd += 1

    autodiff(
        Forward,
        Hop.gf,
        Duplicated(Hop.duplicated, Duplicated(v, result))
    )

	return nothing
end
