#=
Author: Cooper Simpson

SFN optimizer.
=#

# using Zygote: pullback
using Enzyme: ReverseWithPrimal

#=
SFN optimizer struct.
=#
mutable struct SFNOptimizer{T1<:Real, T2<:AbstractFloat, S}
    M::T1 #hessian lipschitz constant
    ϵ::T2 #regularization minimum
    solver::S #search direction solver
    linesearch::Bool #whether to use linsearch
    η::T2 #step-size
    α::T2 #linesearch factor
    atol::T2 #absolute gradient norm tolerance
    rtol::T2 #relative gradient norm tolerance
end

#=
Outer constructor

NOTE: FGQ cant currently handle anything other than Float64

Input:
    dim :: dimension of parameters
    solver :: search direction solver
    M :: hessian lipschitz constant
    ϵ :: regularization minimum
    linsearch :: whether to use linesearch
    η :: step-size in (0,1)
    α :: linsearch factor in (0,1)
    atol :: absolute gradient norm tolerance
    rtol :: relative gradient norm tolerance
=#
function SFNOptimizer(dim::I, solver::Symbol=:KrylovSolver; M::T1=1.0, ϵ::T2=eps(Float64), linesearch::Bool=false, η::T2=1.0, α::T2=0.5, atol::T2=1e-5, rtol::T2=1e-6) where {I<:Integer, T1<:Real, T2<:AbstractFloat}
    #regularization
    @assert (0≤M && 0≤ϵ)

    if linesearch
        @assert (0<α && α<1)
        @assert (0<η && η≤1)
    end

    solver = eval(solver)(dim)

    return SFNOptimizer(M, ϵ, solver, linesearch, η, α, atol, rtol)
end

#=
Repeatedly applies the SFN iteration to minimize the function.

Input:
    opt :: SFNOptimizer
    x :: initialization
    f :: scalar valued function
    itmax :: maximum iterations
    time_limit :: maximum run time
=#
function minimize!(opt::SFNOptimizer, x::S, f::F; itmax::I=1000, time_limit::T2=Inf) where {T1<:AbstractFloat, S<:AbstractVector{T1}, T2, F, I}
    #setup hvp operator
    if typeof(opt.solver) <: KrylovSolver
        power = 2
    else
        power = 1
    end
    Hv = EHvpOperator(f, x, power=power)

    #OLD: Using Zygote
    # function fg!(grads::S, x::S)
        
    #     fval, back = let f=f; pullback(f, x) end
    #     grads .= back(one(fval))[1]

    #     return fval
    # end

    #NEW: Using Enzyme
    function fg!(grads::S, x::S)

        _, fval = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, grads))

        return fval
    end

    #iterate
    stats = iterate!(opt, x, f, fg!, Hv, itmax, time_limit)

    return stats
end

#=
Repeatedly applies the SFN iteration to minimize the function.

Input:
    opt :: SFNOptimizer
    x :: initialization
    f :: scalar valued function
    g! :: inplace gradient function of f
    H :: hvp generator
    itmax :: maximum iterations
    time_limit :: maximum run time
=#
function minimize!(opt::SFNOptimizer, x::S, f::F1, fg!::F2, H::L; itmax::I=1000, time_limit::T=Inf) where {T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, L, I}
    #setup hvp operator
    if (typeof(opt.solver) <: KrylovSolver)
        power = 2
    elseif (typeof(opt.solver) <: ShaleSolver)
        power = 1
    elseif (typeof(opt.solver) <: CraigSolver)
        power = 1
    else
        power = 1
    end
    Hv = LHvpOperator(H, x, power=power)

    #iterate
    stats = iterate!(opt, x, f, fg!, Hv, itmax, time_limit)

    return stats
end

#=
Repeatedly applies the SFN iteration to minimize the function.

Input:
    opt :: SFNOptimizer
    x :: initialization
    f :: scalar valued function
    fg! :: compute f and gradient norm after inplace update of gradient
    Hv :: hvp operator
    itmax :: maximum iterations
    time_limit :: maximum run time
=#
function iterate!(opt::SFNOptimizer, x::S, f::F1, fg!::F2, Hv::H, itmax::I, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, H<:HvpOperator, I}
    #start time
    tic = time_ns()
    
    #stats
    stats = SFNStats(T)
    converged = false
    iterations = 0
    
    #gradient allocation
    grads = similar(x)

    #compute function and gradient
    fval = fg!(grads, x)
    g_norm = norm(grads)

    #estimate regularization
    if opt.linesearch
        ζ = randn(length(x))
        D = norm(ζ)^2

        g2 = similar(grads)
        fg!(g2, x+ζ)

        if any(isnan.(g2)) #this is a bit odd but fixes a particular issue with MISRA1CLS in CUTEst
            opt.M = 1e15
        else
            apply!(ζ, Hv, ζ) 
            ζ .= g2-grads-ζ

            opt.M = min(1e15, 2*norm(ζ)/(D))
        end

        # println("Estimated M: ", opt.M, '\n')

        g2 = nothing #mark for collection
    end

    #compute tolerance
    tol = opt.atol + opt.rtol*g_norm

    #initial stats
    push!(stats.f_seq, fval)
    push!(stats.g_seq, g_norm)

    #iterate
    while iterations<itmax+1
        #check gradient norm
        # println("$g_norm ?< $tol")
        if g_norm <= tol
            converged = true
            break
        end

        #check other exit conditions
        time = elapsed(tic)

        if (time>=time_limit) || (iterations==itmax)
            break
        end

        ##########
        #step

        λ = min(1e15, opt.M*g_norm) #+ opt.ϵ #compute regularization

        step!(opt.solver, stats, Hv, -grads, λ, time_limit-time)

        success = true

        if opt.linesearch
            success = search!(opt, stats, x, opt.solver.p, f, fval, λ)
        else
            x .+= opt.solver.p
        end

        if success == false
            break
        end
        ##########

        #update function and gradient
        fval = fg!(grads, x)
        g_norm = norm(grads)

        #update stats
        push!(stats.f_seq, fval)
        push!(stats.g_seq, g_norm)

        #update hvp operator
        update!(Hv, x)

        #increment
        iterations += 1
    end

    #update stats
    stats.converged = converged
    stats.iterations = iterations
    stats.f_evals += iterations+1
    stats.hvp_evals = Hv.nprod
    stats.run_time = elapsed(tic)

    return stats
end
