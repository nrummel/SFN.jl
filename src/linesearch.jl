#=
Author: Cooper Simpson

R-SFN backtracking line-search procedures.
=#

#=
Regularization line-search.
=#


#=
In place step-size line-search

Input:
    p :: search direction
    x :: current iterate
    f :: scalar valued function
    g_norm :: gradient norm
    α :: float in (0,1)
=#
function search!(p::S, x::S, f::F, g_norm::T; α::T=0.5) where {F, T<:AbstractFloat, S<:AbstractVector{T}}

    @assert (0<α && α<1)

    λ = g_norm

    while true
        r = norm(p-x)^2

        if f(p)-f(x) ≤ -(5/6)*λ*r
            return nothing
        else
            p .*= α
        end
    end

end