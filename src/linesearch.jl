#=
Author: Cooper Simpson

SFN backtracking line-search procedures.
=#

#=
Regularization line-search.
=#


#=
In place step-size line-search

Input:
    x :: search direction
    p :: current iterate
    f :: scalar valued function
    λ :: regularization
    α :: float in (0,1)
=#
function search!(x::S, p::S, f::F, fval::T, λ::T; α::T=0.5) where {F, T<:AbstractFloat, S<:AbstractVector{T}}

    @assert (0<α && α<1)

    dec = sqrt(λ)*(1-3*sqrt(3))/6

    #NOTE: Can we just iteratively update x, is that even that much better?
    #NOTE: This should always exit, but we could also just iterate until r is too small
    iterations = 0

    while true
        iterations += 1

        r = norm(p)^2

        if f(x+p)-fval ≤ dec*r
            break
        else
            p .*= α
        end
    end

    x .+= p

    return iterations

end