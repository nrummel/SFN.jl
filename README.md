# CubicNewton

## TODO
- Implement a new hvp operator using ReverseDiff.jl (or something) instead of Zygote.jl for the backward mode step
- Add appropriate tag instead of nothing for Dual in hvp operator
- Add gradient config stuff to ForwardDiff Dual (and maybe reverse part)
- Should we be passing check_curvature to the cg_lanczos call?
- Only construct hvp operator once and then update it in loops
- Is it possible to only update the partials value of Dual? It seems unlikely since it is immutable
- Switch back to Flux functions for relu, softmax, and logitcrossentropy when NNlibCUDA issue is fixed
- Look into how effective the DataLoaders stuff is, it seems to be wasting a lot of effeciency
- Get the getobs! function working for DataLoaders
- It would be nice if views worked properly, and if one hot arrays could be represented by sparse arrays, but it seems this functionality doesn't exist

## Testing
To test CubicNewton as a package do the following in the REPL:
```julia
using Pkg
Pkg.test(test_args=[<specific tests>])
```

To run experiments you need to be in that folder.
