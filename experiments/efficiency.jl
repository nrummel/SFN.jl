using BenchmarkTools
using JLD2: save

include("functions.jl")

#Setup
# features = collect(50:50:500)
features = [10,20]
labels = ["R-SFN","Newton (AD)", "SFN (AD)", "Newton (Matrix)", "SFN (Matrix)"]
methods = [eval_rsfn, eval_newton, eval_sfn, eval_newton, eval_sfn]
t_arr = zeros(size(features,1), size(labels,1))
m_arr = zeros(size(features,1), size(labels,1))

BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60
t_scale = 1e6
m_scale = (1024)^2

#Benchmark
Threads.@threads for (i, feat) in collect(enumerate(features))
    X, y, f = make_data(feat)
    H = w -> log_hess(w, X, y)

    for (j, method) in enumerate(methods)
        if j<4
            trial = method(f, feat)
        else
            trial = method(f, H, feat)
        end

        t_arr[i,j] = mean(trial.times/t_scale)
        m_arr[i,j] = mean(trial.memory/m_scale)
    end
end

save_data = Dict("labels"=>labels,
                    "features"=>features,
                    "time"=>t_arr,
                    "memory"=>m_arr)

save("efficiency_data.jld2", save_data)
