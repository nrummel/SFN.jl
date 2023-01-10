# Regularized Saddle-Free Newton (R-SFN)

[Cooper Simpson](https://rs-coop.github.io/)

A Julia implementation of the R-SFN algorithm: a second-order method for unconstrained non-convex optimization. To that end, we consider a problem of the following form
$$\min_{\mathbf{x}\in \mathbb{R}^n}f(\mathbf{x})$$
where $f:\mathbb{R}^n\to\mathbb{R}$ is a twice continuously differentiable function. Each iteration applies an update of the following form:
$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)}-\alpha\Big(\big(\nabla^2f(\mathbf{x}^{(k)})\big)^2+\lambda^{(k)}\mathbf{I}\Big)^{-1/2} \nabla f(\mathbf{x}^{(k)})$$
where the regularization term is $\lambda^{(k)}\propto\|\nabla^2f(\mathbf{x}^{(k)})\|$. The matrix inverse square root is computed via a quadrature approximation of the following identity:
$$\mathbf{A}^{-1/2}=\frac{2}{\pi}\int_{0}^{\infty}\big(t^2\mathbf{I}+\mathbf{A}\big)^{-1}\ dt$$
where $\mathbf{A}\in\mathbb{R}^{n\times n}$ is a matrix with strictly positive spectrum, i.e $\sigma(\mathbf{A})\subset\mathbb{R}_{+ +}$.

## License & Citation
All source code is made available under a <insert license>. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See `LICENSE` for the full text.

This repository can be cited using the entry in `CITATION`. For a full list of publications related to R-SFN and influencing this package see [Publications](#publications). If any of these are useful to your own work, please cite them individually.

## Installation

## Environment Setup

## Usage

## Testing
To test the package, run the following command in the REPL:
```julia
using Pkg
Pkg.test(test_args=[<specific tests>])
```

## Publications

### [Regularized Saddle-Free Newton: Saddle Avoidance and Efficient Implementation](https://rs-coop.github.io/projects/research/rsfn)
```
@mastersthesis{rsfn,
	title = {{Regularized Saddle-Free Newton: Saddle Avoidance and Efficient Implementation}},
	author = {Cooper Simpson},
	school = {Dept. of Applied Mathematics, CU Boulder},
	year = {2022},
	type = {{M.S.} Thesis}
}
```
