#=
Author: Cooper Simpson

Line-search procedures.
=#

########################################################

#=
In place SFN step-size line-search

Input:
    x :: current iterate
    p :: search direction
    f :: scalar valued function
    fval :: current function value
    λ :: regularization
    α :: float in (0,1)
=#
function search!(opt::SFNOptimizer, stats::SFNStats, x::S1, f::F, fval::T, g_norm::T) where {F, T<:AbstractFloat, S1<:AbstractVector{T}, S2<:AbstractVector{T}}

    #Setup
    p = opt.solver.p
    p_norm = norm(p)
    success = true

    #Test search direction, select negative gradient if too small
    p_norm = norm(opt.solver.p)

    if p_norm < eps(T)
        stats.status = "Search direction too small"
        return !success
    end
    
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
            stats.status = "Linesearch failed"
            break
        end
    end

    #Update regularization
    # println("Accepted η: ", η)
    opt.M = max(min(1e8, opt.M/η^2), 1e-15)
    # println("Updated M: ", opt.M)

    return success
end

########################################################

#=
In place SFN step-size line-search

Input:
    x :: current iterate
    p :: search direction
    f :: scalar valued function
    fval :: current function value
    λ :: regularization
    α :: float in (0,1)
=#
function search!(opt::ARCOptimizer, stats::SFNStats, x::S1, f::F, fval::T, g_norm::T) where {F, T<:AbstractFloat, S1<:AbstractVector{T}, S2<:AbstractVector{T}}
    
    #Cubic sub-problem
    cubic_subprob = (d) -> begin
        res = similar(b)
        mul!(res, Hv, d)
        return fval + dot(b,d) + 0.5*dot(d, res)
    end

    success = false
    shift_failure = false
    α_new = α
    
    i = findfirst(solver.krylov_solver.converged)

    if i === nothing
        return success
    end

    j = argmin(abs.(opt.α*solver.shifts[i:end]-norm.(solver.krylov_solver.x[i:end]))) + i-1

    while !success && !shift_failure
        stats.f_evals += 1

        ρ = (fval - f(x + solver.krylov_solver.x[j]))/(fval - cubic_subprob(solver.krylov_solver.x[j], fval, grads, Hv))

        #unsuccessful
        if ρ<η1
            α_new = α

            while α_new > γ1*α
                if j == length(shifts)
                    stats.status = "No next shift"
                    shift_failure = true
                    break
                end
                α_new = norm(solver.krylov_solver.x[j+1])/solver.shifts[j+1]
                j += 1
            end
            
        #successful
        else
            success = true

            #step
            solver.p .= solver.krylov_solver.x[j]

            #very successful
            if ρ > η2
                α_new = γ2*α
            else
                α_new = α
            end
        end
    end

    α = α_new

    if α == Inf
        stats.status = "Regularization too large"
    end

    return success
end