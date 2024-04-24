#=
Author: Cooper Simpson

Enzyme AD.
=#

using SFN: HvpOperator
using Enzyme: autodiff, autodiff_deferred, Forward, Reverse, Active, Const, Duplicated, DuplicatedNoNeed

export ehvp, ehvp!, EHvpOperator

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
In-place fast hessian vector product (hvp) function.

Input:
    result :: matvec storage
	f :: scalar valued function
	x :: input to f
	v :: vector
=#
function ehvp!(result::S, f::F, x::S, v::S) where {F, T<:AbstractFloat, S<:AbstractVector{T}}

    bx = similar(x)

    autodiff(
        Forward,
        x -> autodiff_deferred(Reverse, f, Active, x),
        Const,
        DuplicatedNoNeed(Duplicated(x, bx), Duplicated(v, result))
    )

    return nothing
end

#=
In-place hvp operator compatible with Krylov.jl
=#
mutable struct EHvpOperator{F, T<:AbstractFloat, S<:AbstractVector{T}} <: HvpOperator{T}
    x::S
    # duplicated1::Duplicated{S}
    # duplicated2::Duplicated{S}
    gf::F
	nprod::Integer
    power::Integer
end

#=
Base implementations for EHvpOperator
=#
function Base.size(Hv::EHvpOperator)
    n = size(Hv.x, 1)
    
    return (n,n)
end

#=
In place update of EHvpOperator
Input:
	x :: new input to f
=#
function update!(Hv::EHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
    # Hv.duplicated1.val .= x
    # Hv.duplicated1 .= Duplicated(x, similar(x))
    Hv.x .= x

	return nothing
end

#=
Constructor.

Input:
	f :: scalar valued function
	x :: input to f
=#
function EHvpOperator(f::F, x::S; power::Integer=1) where {F, T<:AbstractFloat, S<:AbstractVector{T}}
    
    # duplicated1 = Duplicated(x, similar(x))
    # duplicated2 = Duplicated(similar(x), similar(x))

    # gf = x -> autodiff_deferred(Reverse, f, Active, x)

	return EHvpOperator(x, f, 0, power)
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

    # Hv.duplicated2 .= Duplicated(v, result)
    
    # Hv.duplicated2.val .= v
    # Hv.duplicated2.dval .= result

    # autodiff(
    #     Forward,
    #     # Hv.gf,
    #     x -> autodiff_deferred(Reverse, Hv.gf, Active, x),
    #     Const,
    #     # DuplicatedNoNeed(Hv.duplicated1, Hv.duplicated2)
    #     DuplicatedNoNeed(Duplicated(Hv.x, similar(Hv.x)), Duplicated(v, result))
    # )

    ehvp!(result, Hv.gf, Hv.x, v)

	return nothing
end