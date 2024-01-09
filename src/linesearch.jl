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
function search!(opt::SFNOptimizer, stats::SFNStats, x::S, p::S, f::F, fval::T, λ::T) where {F, T<:AbstractFloat, S<:AbstractVector{T}}

    # println("Reg: ", λ)

    #exit status
    success = true
    
    #increase step-size first
    η = 2.0

    #scale search direction
    p .*= η

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
            # println("Reduction: ", f(x+p)-fval)
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
    # println("η: ", η)
    opt.M = max(min(1e15, opt.M/η^2), 1e-15)
    # println("M: ", opt.M)

    #update iterate
    x .+= p

    return success
end