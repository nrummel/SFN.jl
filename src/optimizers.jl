#=
Author: Cooper Simpson

Cubic Newton optimization functionality in the form of specific optimizers for
each method of solving the subproblem and updating parameters.
=#

#=
Parent type for cubic Newton optimizers.
=#
abstract type CubicNewtonOptimizer end

#=
Minimize the given function according to the subtype update rule.

Input:
    opt :: CubicNewtonOptimizer subtype
    f :: scalar valued function
    grads :: gradient function (defaults to backward mode AD)
    hess :: hessian function (defaults to forward over back AD)
=#
function minimize(opt::CubicNewtonOptimizer, f, x,
                    grads=x -> gradient(f, x), hess=x -> HVPOperator(f, x),
                    itmax=1e3)
    #iterate and update
    for i in 1:opt.itmax
        step(opt, f, x, grads(x), hess(x))
    end
end

#=
Cubic Newton optimizer using shifted Lanczos-CG for solving the sub-problem.
=#
mutable struct ShiftedLanczosCG<:CubicNewtonOptimizer
    σ::Float32 #regularization parameter
    η₁::Float32 #unsuccessful update threshold
    η₂::Float32 #very successful update threshold
    γ₁::Float32 #regularization decrease factor
    γ₂::Float32 #regularization increase factor
    λ::Vector{Float64} #shifts
end
ShiftedLanczosCG(σ=1.0, η₁=0.1, η₂=0.75, γ₁=0.1, γ₂=5.0) = begin
    ShiftedLanczosCG(σ, η₁, η₂, γ₁, γ₂, [10.0^x for x in -15:1:15])
end

#=
Computes an update step according to the shifted Lanczos-CG update rule.

Input:
    f :: scalar valued function
    x :: current iterate
    grads :: function gradients
    hess :: hessian operator
=#
function step!(opt::ShiftedLanczosCG, f, x, grads, hess, last=f(x))
    #solve sub-problem to yield descent direction s and minimum m
    #NOTE: I should maybe be passing check_curvature=true, also good idea to
    #look at other optional arguments.
    (d, stats) = cg_lanczos(hess, -grads, opt.λ)

    #extract indices of shifts that resulted in a positive definite system
    i = findfirst(==(false), stats.indefinite)

    num_shifts = size(opt.λ)
    while i <= num_shifts
        #update and evaluate
        x .+= d[i]
        ρ = (f(x) - last)/_quadratic_eval(d[i], grads, hess)

        #unsuccessful, so consider other shifts
        if ρ<opt.η₁
            x .-= s #undo parameter update

            σ₀ = opt.σ
            while opt.σ > opt.γ₁*σ₀
                i < num_shifts ? i+=1 : return false #try next shift
                opt.σ = norm(d[i])/opt.λ[i] #decrease regularization parameter
            end

        #medium success, do nothing

        #very successful
        elseif ρ>opt.η₂
            opt.σ *= opt.γ₂ #increase regularization parameter
        end
    end

    return true
end

#=
Cubic Newton optimizer solving the subproblem via an eigenvalue problem.
=#
mutable struct Eigen <: CubicNewtonOptimizer
    σ::Float32 #regularization parameter
    η₁::Float32 #unsuccessful update threshold
    η₂::Float32 #very successful update threshold
    γ₁::Float32 #regularization decrease factor
    γ₂::Float32 #regularization increase factor
    dummy::Bool
end
Eigen(σ=1.0, η₁=0.1, η₂=0.75, γ₁=0.1, γ₂=5.0) = begin
    Eigen(σ, η₁, η₂, γ₁, γ₂, true)
end

#=
Computes an update step according to the Eigen update rule.

Input:
    f :: scalar valued function
    x :: current iterate
    grads :: function gradients
    hess :: hessian operator
=#
function step!(opt::Eigen, f, x, grads, hess)
    #solve sub-problem to yield descent direction s and minimum m

    #evaluate descent direction, adapt hyperparameters, update parameters
    x .-= s
    ρ = (f(x) - loss)/m

    #bad update
    if ρ<opt.η₁
        opt.σ *= opt.γ₁
        params .+= s #undo parameter update
    #great update
    elseif ρ>=opt.η₂
        opt.σ *= opt.γ₂
    end

    return true
end
