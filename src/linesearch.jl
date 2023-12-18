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
    success = true
    
    #increase step-size first
    if searcher.η >= 1.0
        searcher.η *= 2
        p .*= searcher.η
    else
        searcher.η = 1.0
    end

    #search direction norm
    p_norm = norm(p)

    if p_norm < eps(T)
        stats.status = "search direction too small"
        return false
    end
    
    #target decrement
    dec = p_norm^2*sqrt(λ)*(1-3*sqrt(3))/6

    #NOTE: Can we just iteratively update x, is that even that much better?
    while true
        stats.f_evals += 1

        if f(x+p)-fval ≤ dec
            break
        else
            searcher.η *= searcher.α #reduce step-size
            p .*= searcher.α #scale search direction
            dec *= searcher.α^2 #scale decrement
        end

        #check step-size
        if (searcher.η < eps(T)) || (isnan(searcher.η))
            success = false
            stats.status = "Linesearch failed"
            break
        end
    end

    x .+= p

    return success
end

# function search!(searcher::SFNLineSearcher, stats::SFNStats, x::S, p::S, f::F, fval::T, λ::T) where {F, T<:AbstractFloat, S<:AbstractVector{T}}

#     #exit status
#     success = true
    
#     #increase step-size first
#     if searcher.η >= 1.0
#         searcher.η *= 2
#         p .*= searcher.η
#     else
#         searcher.η = 1.0
#     end

#     #search direction norm
#     p_norm = norm(p)

#     if p_norm < eps(T)
#         stats.status = "search direction too small"
#         return false
#     end
    
#     #target decrement
#     dec = p_norm^2*sqrt(λ)*(1-3*sqrt(3))/6
#     prev = Inf

#     #NOTE: Can we just iteratively update x, is that even that much better?
#     while true
#         stats.f_evals += 1

#         reduction = f(x+p)-fval

#         println(reduction, " ", searcher.η)

#         if reduction ≤ dec
#             break
#         elseif reduction > prev && prev < 0.0
#             searcher.η /= searcher.α
#             p ./= searcher.α
#             break
#         else
#             searcher.η *= searcher.α #reduce step-size
#             p .*= searcher.α #scale search direction
#             dec *= searcher.α^2 #scale decrement
#         end

#         #check step-size
#         if (searcher.η < eps(T)) || (isnan(searcher.η))
#             if prev < 0.0
#                 searcher.η /= searcher.α
#                 p ./= searcher.α
#                 break
#             end

#             success = false
#             stats.status = "Linesearch failed"
#             break
#         end

#         prev = reduction
#     end

#     x .+= p

#     return success
# end