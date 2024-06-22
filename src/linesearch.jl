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
function search!(opt::SFNOptimizer, stats::SFNStats, x::S1, p::S2, p_norm::T, f::F, fval::T, λ::T) where {F, T<:AbstractFloat, S1<:AbstractVector{T}, S2<:AbstractVector{T}}

    #Exit status
    success = true
    
    #Increase step-size
    η = 2.0

    #Scale search direction and norm
    p .*= η
    p_norm *= η 
    
    #Target decrement
    dec = p_norm^2*sqrt(λ)*(1-3*sqrt(3))/6

    #NOTE: Can we just iteratively update x, is that even that much better?
    while true
        stats.f_evals += 1

        if f(x+p)-fval ≤ dec
            # println("Reduction: ", f(x+p)-fval, " Dec: ", dec)
            break
        else
            η *= opt.α #reduce step-size
            p .*= opt.α #scale search direction
            dec *= opt.α^2 #scale decrement
        end

        #Check step-size
        if η < eps(T)
            success = false
            stats.status = "linesearch failed"
            break
        end
    end

    #Update regularization
    # println("Accepted η: ", η)
    opt.M = max(min(1e8, opt.M/η^2), 1e-8)
    # println("Updated M: ", opt.M)

    return success
end