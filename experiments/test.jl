### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 916d1d9c-b7b6-4a02-b810-f792787a4314
begin
	import Pkg
	Pkg.activate(".")
	include("functions.jl")
	using RSFN
end

# ╔═╡ 839bf7b8-b14d-4386-9527-acedb73b8f97
dim = 10

# ╔═╡ a7590d94-13a3-4544-84bc-f2a4364c1871
begin
	x1 = zeros(dim)
	newton!(x1, rosenbrock, itmax=15)
end

# ╔═╡ 3ec719cb-a29a-4833-9aff-f80de70ef126
rosenbrock(x1)

# ╔═╡ a0737f74-127c-4c2d-8076-e1ec08da37d7
begin
	x2 = zeros(dim)
	opt = RSFNOptimizer(dim, M=0, ϵ=0.0, quad_order=200)
	@time minimize!(opt, x2, rosenbrock, itmax=15)
end

# ╔═╡ 82f19623-bc3d-44a4-a1be-a0e20627c2ea
rosenbrock(x2)

# ╔═╡ Cell order:
# ╠═916d1d9c-b7b6-4a02-b810-f792787a4314
# ╠═839bf7b8-b14d-4386-9527-acedb73b8f97
# ╠═a7590d94-13a3-4544-84bc-f2a4364c1871
# ╠═3ec719cb-a29a-4833-9aff-f80de70ef126
# ╠═a0737f74-127c-4c2d-8076-e1ec08da37d7
# ╠═82f19623-bc3d-44a4-a1be-a0e20627c2ea
