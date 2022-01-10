### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ 912ce392-7231-11ec-0653-43f50494e143
begin
	using Pkg
	Pkg.activate("../.")
	using CubicNewton
end

# ╔═╡ b614d1e2-28b6-4e5e-90c0-9e2de8973dc2
function rosenbrock(x)
	res = 0.0
	for i = 1:size(x,1)-1
		res += 100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
	end
	return res
end

# ╔═╡ 59f5b64f-c8ca-4f1a-9ba0-7440358d28dc
begin
	x1 = [0.0, 3.0]
	minimize!(ShiftedLanczosCG(), rosenbrock, x1, itmax=12)
end

# ╔═╡ c786f50e-6c0a-4d73-ad9c-c3323a8a1176
x1

# ╔═╡ 40381e39-cf2d-4a6d-8fab-04de86dc2107
begin
	x2 = randn(10)
	minimize!(ShiftedLanczosCG(), rosenbrock, x2, itmax=10)
end

# ╔═╡ b1c9bb2f-9796-4c10-ac4d-0dd88b80800f
x2

# ╔═╡ Cell order:
# ╠═912ce392-7231-11ec-0653-43f50494e143
# ╠═b614d1e2-28b6-4e5e-90c0-9e2de8973dc2
# ╠═59f5b64f-c8ca-4f1a-9ba0-7440358d28dc
# ╠═c786f50e-6c0a-4d73-ad9c-c3323a8a1176
# ╠═40381e39-cf2d-4a6d-8fab-04de86dc2107
# ╠═b1c9bb2f-9796-4c10-ac4d-0dd88b80800f
