# Regularized Saddle-Free Newton

## TODO
- HVP operator:
  - Should probably only require one of zygote or reversediff and use that in the minimize! function
  - Check if I am doing Tags correctly for Dual, and also add Tag to Zygote hvp operator
  - Add gradient config stuff to ForwardDiff Dual (and maybe reverse part)
- Flux integration:
  - It would be nice if views worked properly, and if one hot arrays could be represented by sparse arrays, but it seems this functionality doesn't exist

## Testing
To test R-SFN as a package do the following in the REPL:
```julia
using Pkg
Pkg.test(test_args=[<specific tests>])
```

To run experiments you need to be in that folder.
