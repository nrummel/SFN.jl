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

#     #NOTE: Can we just iteratively update x, is that even that much better?
#     while true
#         stats.f_evals += 1

#         if f(x+p)-fval ≤ dec
#             break
#         else
#             searcher.η *= searcher.α #reduce step-size
#             p .*= searcher.α #scale search direction
#             dec *= searcher.α^2 #scale decrement
#         end

#         #check step-size
#         if (searcher.η < eps(T)) || (isnan(searcher.η))
#             success = false
#             stats.status = "Linesearch failed"
#             break
#         end
#     end

#     x .+= p

#     return success
# end

function search!(opt::SFNOptimizer, stats::SFNStats, x::S, p::S, f::F, fval::T, λ::T) where {F, T<:AbstractFloat, S<:AbstractVector{T}}

    #exit status
    success = true
    
    #increase step-size first
    η = 1.0

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
            η *= opt.α #reduce step-size
            p .*= opt.α #scale search direction
            dec *= opt.α^2 #scale decrement
        end

        #check step-size
        if η < eps(T)
            success = false
            stats.status = "linesearch failed"
            break
        end
    end

    #update regularization
    opt.M = 1/η^2

    #update iterate
    x .+= p

    return success
end

# function search!(opt::SFNOptimizer, stats::SFNStats, x::S, p::S, fg!::F, fval::T, λ::T) where {F, T<:AbstractFloat, S<:AbstractVector{T}}

#     #exit status
#     success = true

#     #allocate gradient space
#     g_next = similar(x)
    
#     #increase step-size first
#     η = 1.0

#     #search direction norm
#     p_norm = norm(p)

#     if p_norm < eps(T)
#         stats.status = "Search direction too small"
#         return false
#     end
    
#     #target decrement
#     dec1 = p_norm^2*sqrt(λ)*(1-3*sqrt(3))/6
#     dec2 = (3/2)*sqrt(λ)*p_norm

#     #NOTE: Can we just iteratively update x, is that even that much better?
#     while true
#         stats.f_evals += 1

#         f_next = fg!(g_next, x+p)

#         # if f(x+p)-fval ≤ dec
#         if (f_next-fval ≤ dec1) || (norm(g_next) ≤ dec2)
#             break
#         else
#             η *= opt.α #reduce step-size
#             p .*= opt.α #scale search direction
#             dec1 *= opt.α^2 #scale decrement
#             dec2 *= opt.α
#         end

#         #check step-size
#         if η < eps(T)
#             success = false
#             stats.status = "Linesearch failed"
#             break
#         end
#     end

#     #update regularization
#     opt.M = 1/η^2

#     #update iterate
#     x .+= p

#     return success
# end