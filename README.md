# CubicNewton

## TODO
- Update the hvp operator according to information from that GitHub question on SparseDiffTools.jl
- Implement a new hvp operator using ReverseDiff.jl instead of Zygote.jl for the backward mode step

## Testing
'''julia
julia> Pkg.test(test_args=[<specific tests>])
'''
