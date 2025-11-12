"""
    Arrowhead

A Julia package for solving large-scale arrowhead-structured linear systems
using Schur complement methods on multiple GPUs with CUDSS.

The package implements the augmented factorization approach with BiCGStab
iterative refinement for distributed stochastic optimization problems.
"""
module Arrowhead

using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR
using CUDSS
using LinearAlgebra
using SparseArrays

# Include source files
include("types.jl")
include("cusolver_utils.jl")
include("schur.jl")
include("matvec.jl")
include("preconditioner.jl")
include("factorize.jl")
include("bicgstab.jl")
include("solver.jl")

# Export main types
export ArrowheadSystem, ArrowheadSolver, ArrowheadVector
export ArrowheadWorkspace

# Export main functions (LinearAlgebra-style API)
# Note: factorize is extended from LinearAlgebra, not exported
export solve!

# Export utility functions
export matvec

# Backward compatibility
export ArrowheadFactors

end # module

