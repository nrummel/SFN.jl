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
    maxIter :: maximum iterations (optional)
=#
function minimize!(opt::CubicNewtonOptimizer, f, x;
                    grads=x -> gradient(f, x)[1], hess=x -> HvpOperator(f, x),
                    maxIter=1e3)
    #iterate and update
    for i in 1:maxIter
        step!(opt, f, x, grads(x), hess(x))
    end
end

#=
Cubic Newton optimizer using shifted Lanczos-CG for solving the sub-problem.
=#
Base.@kwdef mutable struct ShiftedLanczosCG<:CubicNewtonOptimizer
    σ::Float32 = 1.0 #regularization parameter
    η₁::Float32 = 0.1 #unsuccessful update threshold
    η₂::Float32 = 0.75 #very successful update threshold
    γ₁::Float32 = 0.1 #regularization decrease factor
    γ₂::Float32 = 5.0 #regularization increase factor
    λ::Vector{Float32} = [10.0^x for x in -15:1:15] #shifts
end

#=
Computes an update step according to the shifted Lanczos-CG update rule.

Input:
    f :: scalar valued function
    x :: current iterate
    grads :: function gradients
    hess :: hessian operator
    last :: current function value at x
=#
function step!(opt::ShiftedLanczosCG, f, x, grads, hess, last=f(x))
    #solve sub-problem to yield descent direction d
    n = size(grads, 1)
    solver = CgLanczosShiftSolver(n, n, size(opt.λ, 1), Vector{Float32})
    cg_lanczos!(solver, hess, -grads, opt.λ, check_curvature=true)

    #extract indices of shifts that resulted in a positive definite system
    i = findfirst(==(false), solver.stats.indefinite)

    #extract solutions
    d = solver.x

    numShifts = size(opt.λ, 1)
    while i <= numShifts
        #update and evaluate
        x .+= d[i]
        ρ = (f(x) - last)/_quadratic_eval(d[i], grads, hess)

        #unsuccessful, so consider other shifts
        if ρ<opt.η₁
            x .-= d[i] #undo parameter update

            σ₀ = opt.σ
            while opt.σ > opt.γ₁*σ₀
                i < numShifts ? i+=1 : return false #try next shift
                opt.σ = norm(d[i])/opt.λ[i] #decrease regularization parameter
            end
        #medium success, do nothing
        #very successful
        elseif ρ>opt.η₂
            opt.σ *= opt.γ₂ #increase regularization parameter
            break
        end
    end

    return true
end

#=
Cubic Newton optimizer solving the subproblem via an eigenvalue problem.
=#
Base.@kwdef mutable struct Eigen <: CubicNewtonOptimizer
    σ::Float32 = 1.0 #regularization parameter
    η₁::Float32 = 0.1 #unsuccessful update threshold
    η₂::Float32 = 0.75 #very successful update threshold
    γ₁::Float32 = 0.1 #regularization decrease factor
    γ₂::Float32 = 5.0 #regularization increase factor
end

#=
Computes an update step according to the Eigen update rule.

Input:
    f :: scalar valued function
    x :: current iterate
    grads :: function gradients
    hess :: hessian operator
=#
function step!(opt::Eigen, f, x, grads, hess, last=f(x))
    #solve sub-problem to yield descent direction d
    #d = eignevalues...

    #update and evaluate
    x .+=
    ρ = (f(x) - last)/_quadratic_eval(d, grads, hess)

    #unsuccessful, so consider other shifts
    if ρ<opt.η₁
        x .-= s #undo parameter update

        σ₀ = opt.σ
        while opt.σ > opt.γ₁*σ₀
            opt.σ = norm(d[i])/opt.λ[i] #decrease regularization parameter
        end

    #medium success, do nothing

    #very successful
    elseif ρ>opt.η₂
        opt.σ *= opt.γ₂ #increase regularization parameter
    end

    return true
end
