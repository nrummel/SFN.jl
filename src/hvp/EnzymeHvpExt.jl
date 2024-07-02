#=
Author: Cooper Simpson

Enzyme AD.
=#

using SFN: HvpOperator
using Enzyme: make_zero, make_zero!, autodiff, autodiff_deferred, Forward, Reverse, Active, Const, Duplicated, DuplicatedNoNeed

export ehvp, ehvp!, EHvpOperator

#=
Fast hessian vector product (hvp) function.

Input:
	f :: scalar valued function
	x :: input to f
	v :: vector
=#
function ehvp(f::F, x::S, v::S) where {F, T<:AbstractFloat, S<:AbstractVector{T}}

    res = similar(x)
    ehvp!(res, f, x, v)

    return res
end

#=
In-place fast hessian vector product (hvp) function.

Input:
    res :: matvec storage
	f :: scalar valued function
	x :: input to f
	v :: vector
=#
function ehvp!(res::S, f::F, x::S, v::S) where {F, T<:AbstractFloat, S<:AbstractVector{T}}

    make_zero!(res)
    grad = make_zero(x)

    autodiff(
        Forward,
        d -> autodiff_deferred(Reverse, f, Active, d),
        Const,
        DuplicatedNoNeed(Duplicated(x, grad), Duplicated(v, res))
    )

    return nothing
end

#=
In-place hvp operator compatible with Krylov.jl
=#
mutable struct EHvpOperator{F, T<:AbstractFloat, S<:AbstractVector{T}, I<:Integer} <: HvpOperator{T}
    duplicated::DuplicatedNoNeed{Duplicated{S}}
    const f::F
	nprod::I
    const power::I
end

#=
Base implementations for EHvpOperator
=#
function Base.size(Hv::EHvpOperator)
    n = size(Hv.duplicated.val.val, 1)
    
    return (n,n)
end

#=
In place update of EHvpOperator
Input:
	x :: new input to f
=#
function update!(Hv::EHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
    
    Hv.duplicated.val.val .= x

	return nothing
end

#=
Constructor.

Input:
	f :: scalar valued function
	x :: input to f
=#
function EHvpOperator(f::F, x::S; power::Integer=1) where {F, T<:AbstractFloat, S<:AbstractVector{T}}
    
    duplicated1 = Duplicated(x, similar(x))
    duplicated2 = Duplicated(similar(x), similar(x))

    return EHvpOperator(DuplicatedNoNeed(duplicated1, duplicated2), f, 0, power)
end

#=
Inplace matrix vector multiplcation with EHvpOperator.

Input:
	res :: matvec storage
	Hv :: EHvpOperator
	v :: rhs vector
=#
function apply!(res::S, Hv::EHvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	Hv.nprod += 1

    make_zero!(Hv.duplicated.val.dval)

    Hv.duplicated.dval.val .= v
    make_zero!(Hv.duplicated.dval.dval)

    autodiff(
        Forward,
        d -> autodiff_deferred(Reverse, Hv.f, Active, d),
        Const,
        Hv.duplicated
    )

    res .= Hv.duplicated.dval.dval

	return nothing
end