#=
Author: Cooper Simpson

SFN backtracking line-search procedures.
=#

#=
In place step-size line-search

Input:
    x :: current iterate
    p :: search direction
    f :: scalar valued function
    fval :: current function value
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

#=
SFN line-search struct.
=#
mutable struct SFNLineSearcher{T<:AbstractFloat, I<:Integer}
    η::T
    α::T
    iterations::I
end

#=
Outer constructor.
=#
function SFNLineSearcher(;η::T=1.0, α::T=0.5) where {T<:AbstractFloat}
    
    @assert η>0
    @assert (0<α && α<1)

    return SFNLineSearcher(η, α, 0)
end

#=
In place line-search procedure.

Input:
    searcher :: SFNLineSearcher
    stats :: SFNStats
    x :: current iterate
    p :: search direction
    f :: scalar valued function
    fval :: current function value
    λ :: regularization
=#
function search!(searcher::SFNLineSearcher, stats::SFNStats, x::S, p::S, f::F, fval::T, λ::T) where {F, T<:AbstractFloat, S<:AbstractVector{T}}

    #exit status
    status = true
    
    #increase step-size first
    searcher.η *= 2
    p .*= searcher.η
    # searcher.η = 1.0
    
    dec = sqrt(λ)*(1-3*sqrt(3))/6

    #NOTE: Can we just iteratively update x, is that even that much better?
    #NOTE: This should always exit, but we could also just iterate until r is too small

    while true
        r = norm(p)^2

        if (r < eps(T)) || (isnan(r))
            status = false
            break
        end

        stats.f_evals += 1

        if f(x+p)-fval ≤ dec*r
            break
        else
            searcher.η *= searcher.α
            p .*= searcher.η
            # p .*= searcher.α
        end
    end

    x .+= p

    return status
end