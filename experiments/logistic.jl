### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 302f36c5-10ac-4377-8b07-2c7eb9ba66f2
begin
	using Pkg
	Pkg.activate(".")
	using NNlib: logsigmoid
	using MLJ: make_regression
end

# ╔═╡ d16c9dee-fbbe-11ec-1ae0-bddf1ced22b9
function logistic_loss(x, y)
	return -sum(logsigmoid.(x.*y))
end

# ╔═╡ b17057ac-d9c2-4a9e-a85e-d0a1b240be9c
begin
	num_obs = 100
	num_feat = 5
	X, y = make_regression(num_obs, num_feat; rng=1, as_table=false, binary=false)
	y = map(x -> x<0 ? -1 : 1, y);
end

# ╔═╡ 998604ab-3baf-409e-8589-6097be86cb31
function accuracy(θ)
	return acc = sum(map(x -> x<0 ? -1 : 1, X*θ) .== y)/num_obs
end

# ╔═╡ c0a25367-bb5a-4f0b-9c60-0c067bd25aae
begin
	w = zeros(num_feat)
	f(w) = logistic_loss(X*w, y)
	accuracy(w)
end

# ╔═╡ a8a0e62b-b593-475d-8f0f-ec47cfad2caa
begin
	using RSFN
	opt = RSFNOptimizer(num_feat, M=0, ϵ=0.0, quad_order=100)
	minimize!(opt, w, f, itmax=100)
end

# ╔═╡ 1eeb0e61-8ad1-44c6-811a-9859ae563321
accuracy(w)

# ╔═╡ Cell order:
# ╠═302f36c5-10ac-4377-8b07-2c7eb9ba66f2
# ╠═d16c9dee-fbbe-11ec-1ae0-bddf1ced22b9
# ╠═b17057ac-d9c2-4a9e-a85e-d0a1b240be9c
# ╠═998604ab-3baf-409e-8589-6097be86cb31
# ╠═c0a25367-bb5a-4f0b-9c60-0c067bd25aae
# ╠═a8a0e62b-b593-475d-8f0f-ec47cfad2caa
# ╠═1eeb0e61-8ad1-44c6-811a-9859ae563321
