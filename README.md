# CubicNewton

## TODO
- Update the hvp operator according to information from that GitHub question on SparseDiffTools.jl
- Implement a new hvp operator using ReverseDiff.jl instead of Zygote.jl for the backward mode step

## Testing
To test CubicNewton as a package do the following in the REPL:
```julia
using Pkg
Pkg.test(test_args=[<specific tests>])
```

To run experiments you need to be in that folder.
