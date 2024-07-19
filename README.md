# QuasiNewton

A collection of Newton-type optimization algorithms.

### Authors: [Cooper Simpson](https://rs-coop.github.io/)

## License & Citation
All source code is made available under an MIT license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See `LICENSE` for the full text.

This repository can be cited using the GitHub action in the sidebar, or using the metadata in `CITATION.cff`. See [Publications](#publications) for a full list of publications related to R-SFN and influencing this package. If any of these are useful to your own work, please cite them individually.

## Installation
This package can be installed just like any other Julia package. From the terminal, after starting the Julia REPL, run the following:
```julia
using Pkg
Pkg.add("QuasiNewton")
```
This will install the package and its direct dependencies, but in order to use the package you must install one of the following sets of packages for automatic differentiation (AD):
- `Enzyme.jl`
- `ReverseDiff.jl` and `ForwardDiff.jl`
- `Zygote.jl` and `ForwardDiff.jl`

### Testing
To test the package, run the following command in the REPL:
```julia
using Pkg
Pkg.test(test_args=["optional specific tests"])
```

## Usage
Load the package as usual:
```julia
using QuasiNewton
```
which will export the struct `RSFNOptimizer` and the `minimize!` function. Then load your AD packages, which will export a subtype of `HvpOperator`. Say you load Enzyme:
```julia
using Enzyme
```
then the `EHvpOperator` will be available.

Let's look at a two dimensional Rosenbrock example:
```julia
function rosenbrock(x)

	res = 0.0
	for i = 1:size(x,1)-1
		res += 100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
	end

	return res

end

x = [0.0, 0.0]

opt = RSFNOptimizer(size(x,1))

minimize!(opt, x, rosenbrock, itmax=10)
```

## Publications

### [Regularized Saddle-Free Newton: Saddle Avoidance and Efficient Implementation](https://rs-coop.github.io/projects/research/rsfn)
```bibtex
@mastersthesis{rsfn,
	title = {{Regularized Saddle-Free Newton: Saddle Avoidance and Efficient Implementation}},
	author = {Cooper Simpson},
	school = {Dept. of Applied Mathematics, CU Boulder},
	year = {2022},
	type = {{M.S.} Thesis}
}
```
