### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 302f36c5-10ac-4377-8b07-2c7eb9ba66f2
begin
	using Pkg
	Pkg.activate(".")
	using NNlib: logsigmoid, sigmoid
	using MLJ: make_regression
	using RSFN
	import LinearAlgebra as LA
	using BenchmarkTools
	using Statistics
end

# ╔═╡ eeed0be7-704e-4c3b-ac55-e3617ffb93b5
include("functions.jl")

# ╔═╡ d16c9dee-fbbe-11ec-1ae0-bddf1ced22b9
function logistic_loss(x, y)
	return -sum(logsigmoid.(x.*y))
end

# ╔═╡ 998604ab-3baf-409e-8589-6097be86cb31
function accuracy(θ, X, y)
	return acc = sum(map(x -> x<0 ? -1 : 1, X*θ) .== y)/size(X,1)
end

# ╔═╡ b17057ac-d9c2-4a9e-a85e-d0a1b240be9c
function make_data(num_feat)
	num_obs = num_feat*10
	X, y = make_regression(num_obs, num_feat; rng=57, as_table=false, binary=false, 							intercept=false, noise=1)
	y = map(x -> x<0 ? -1 : 1, y)

	f(w) = logistic_loss(X*w, y)

	return X, y, f
end

# ╔═╡ 52e78093-e179-43af-8163-0ba8c379cb33
function log_hess(w, X, y)
	u = sigmoid.(-y.*(X*w))
	S = LA.Diagonal(u.*(ones(size(u,1))-u))
	return transpose(X)*S*X + 1e-6*LA.I(size(X,2))
end

# ╔═╡ cd06b37b-def0-4217-a38d-94785eb648cb
begin
	features = [10, 20, 30]
	labels = ["R-SFN","Newton (AD)", "Newton (Matrix)", "SFN (AD)", "SFN (Matrix)"]
	m_arr = zeros(size(features,1),size(labels,1))
	std_arr = zeros(size(features,1),size(labels,1))
end

# ╔═╡ e10a5576-c726-4664-bde6-b1317872b7f5
begin
	using Plots
	plot(features, m_arr, yerr=std_arr,
		xlabel="Problem Dimension",
		ylabel="Execution Time (ms)",
		w=3,
		label=reshape(labels, 1, size(labels,1)),
		legend=:topleft)
end

# ╔═╡ 6da06660-0d2b-4b38-9813-475dfa681f92
begin
	BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
	# BenchmarkTools.DEFAULT_PARAMETERS.seconds = 600
	scale = 1e6
end

# ╔═╡ 889e968c-5fba-4dc2-8be7-5286fb48786b
begin
	for (i, feat) in enumerate(features)
		X, y, f = make_data(feat)
		
		trial = eval_rsfn(f, feat)
		times = trial.times/scale
		
		m_arr[i,1] = mean(times)
		std_arr[i,1] = std(times)

		println(size(times))
	end
end

# ╔═╡ 12131288-8f6b-4175-a57b-adbeabc7bb88
begin
	for (i, feat) in enumerate(features)
		X, y, f = make_data(feat)
		
		trial = eval_newton(f, feat)
		times = trial.times/scale
		
		m_arr[i,2] = mean(times)
		std_arr[i,2] = std(times)

		println(size(times))
	end
end

# ╔═╡ f38f84c6-2019-4496-a61a-258eb8f5983c
begin
	for (i, feat) in enumerate(features)
		X, y, f = make_data(feat)
		
		trial = eval_newton(f, w -> log_hess(w, X, y), feat)
		times = trial.times/scale
		
		m_arr[i,3] = mean(times)
		std_arr[i,3] = std(times)

		println(size(times))
	end
end

# ╔═╡ b0962577-6fc3-42f7-a01b-dc95bb4b5155
begin
	for (i, feat) in enumerate(features)
		X, y, f = make_data(feat)
		
		trial = eval_sfn(f, feat)
		times = trial.times/scale
		
		m_arr[i,4] = mean(times)
		std_arr[i,4] = std(times)

		println(size(times))
	end
end

# ╔═╡ d595d778-518e-478b-a41e-52ef8c09ed0c
begin
	for (i, feat) in enumerate(features)
		X, y, f = make_data(feat)
		
		trial = eval_sfn(f, w -> log_hess(w, X, y), feat)
		times = trial.times/scale
		
		m_arr[i,5] = mean(times)
		std_arr[i,5] = std(times)

		println(size(times))
	end
end

# ╔═╡ Cell order:
# ╠═302f36c5-10ac-4377-8b07-2c7eb9ba66f2
# ╠═d16c9dee-fbbe-11ec-1ae0-bddf1ced22b9
# ╠═998604ab-3baf-409e-8589-6097be86cb31
# ╠═b17057ac-d9c2-4a9e-a85e-d0a1b240be9c
# ╠═eeed0be7-704e-4c3b-ac55-e3617ffb93b5
# ╠═52e78093-e179-43af-8163-0ba8c379cb33
# ╠═cd06b37b-def0-4217-a38d-94785eb648cb
# ╠═6da06660-0d2b-4b38-9813-475dfa681f92
# ╠═889e968c-5fba-4dc2-8be7-5286fb48786b
# ╠═12131288-8f6b-4175-a57b-adbeabc7bb88
# ╠═f38f84c6-2019-4496-a61a-258eb8f5983c
# ╠═b0962577-6fc3-42f7-a01b-dc95bb4b5155
# ╠═d595d778-518e-478b-a41e-52ef8c09ed0c
# ╠═e10a5576-c726-4664-bde6-b1317872b7f5
