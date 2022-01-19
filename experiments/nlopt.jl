using JuMP
using EAGO
using LinearAlgebra

n₀ = 28*28
n₃ = 10
A = diagm(-1 => [1])

model = Model(EAGO.Optimizer)

@variable(model, x[i=1:2] >= 10)
@NLconstraint(model, n₀*x[1] + x[1]*x[2] + n₃*x[2] >= 60000.0)
@NLobjective(model, Min, n₀*x[1] + x[1]*x[2] + n₃*x[2])

optimize!(model)
print(model)
println(value.(x))
