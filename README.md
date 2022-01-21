# CubicNewton

## TODO
- Implement a new hvp operator using ReverseDiff.jl (or something) instead of Zygote.jl for the backward mode step
- Add appropriate tag instead of nothing for Dual in hvp operator
- Add gradient config stuff to ForwardDiff Dual (and maybe reverse part)

## Testing
To test CubicNewton as a package do the following in the REPL:
```julia
using Pkg
Pkg.test(test_args=[<specific tests>])
```

To run experiments you need to be in that folder.
