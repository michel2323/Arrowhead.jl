# Arrowhead.jl Implementation Summary

## Overview

This document provides a technical summary of the Arrowhead.jl implementation, which solves arrowhead-structured linear systems using GPU-batched Schur complement methods on multiple GPUs with Krylov.jl iterative solvers.

## Package Structure

```
Arrowhead.jl/
├── Project.toml              # Package dependencies and metadata
├── README.md                 # User documentation
├── IMPLEMENTATION.md         # Technical implementation details
├── LICENSE                   # MIT License
├── .gitignore               # Git ignore patterns
├── src/
│   ├── Arrowhead.jl         # Main module file
│   ├── types.jl             # Core data structures
│   ├── cusolver_utils.jl    # cuSOLVER wrappers for dense LU
│   ├── schur.jl             # GPU-batched Schur complement computation
│   ├── matvec.jl            # Matrix-vector products
│   ├── preconditioner.jl    # Preconditioner solve
│   ├── factorize.jl         # System factorization
│   ├── bicgstab.jl          # BiCGStab solver (Krylov.jl integration)
│   └── solver.jl            # LinearAlgebra interface (backslash, ldiv!)
├── test/
│   └── runtests.jl          # Test suite
└── examples/
    └── simple_arrowhead.jl  # Usage example
```

## Core Components

### 1. Data Structures (`types.jl`)

#### ArrowheadSystem{T,INT,BCT}
Represents the block-structured system:
- `K0`: First-stage sparse matrix (n0 × n0)
- `K_scenarios`: Vector of scenario sparse matrices
- `B_coupling`: Vector of dense coupling matrices  (BCT = CuMatrix or CuSparseMatrixCSR)
- `device_map`: GPU assignment for scenarios
- `n0`, `n_scenarios`, `scenario_dims`: Dimension information
- `structure`, `view`: Matrix properties for CUDSS ("SPD", "S", "H", "G" and 'U', 'L', 'F')

#### ArrowheadVector{T,VT} <: AbstractVector{T}
Distributed vector implementing full AbstractVector interface:
- `z0`: First-stage variables (CuVector)
- `z_scenarios`: Vector of scenario variables
- **Implements**:  `length`, `size`, `eltype`, `getindex`, `setindex!`, `similar`, `copyto!`, `fill!`
- **Vector operations**: `dot`, `norm`, `axpy!`, `axpby!`, `copy!`
- **Critical**: All operations avoid scalar indexing for GPU performance

#### ArrowheadWorkspace{T}
Preallocated workspace for in-place operations:
- **BiCGStab vectors**: `x`, `r`, `r_tilde`, `p`, `v`, `s`, `t`, `y`, `z`, `temp`
- **Preconditioner vectors**: `r_aug_gpus` (per-GPU augmented vectors), `r_tilde_0` (first-stage)
- **Krylov workspace**: `krylov_workspace` (lazily allocated `Krylov.BicgstabWorkspace`)

#### ArrowheadSolver{T,INT,BCT} <: Factorization{T}
Main solver type with factorization data:
- `system`: The ArrowheadSystem being solved
- `gpu_solvers`: Dict mapping GPU ID → CUDSS solver for augmented system
- `scenario_to_gpu`, `gpu_to_scenarios`: GPU assignment mappings
- `schur_complements`: Vector of per-GPU Schur complements
- `C_aggregated`: Dense aggregated global Schur complement
- `C_factors`: cuSOLVER LU factorization of C
- `workspace`: ArrowheadWorkspace with preallocated vectors
- `tol`, `maxiter`, `verbose`: Solver parameters
- `factorized`: Boolean flag indicating factorization status
- **Note**: `ArrowheadFactors` is aliased to `ArrowheadSolver` for backward compatibility

### 2. cuSOLVER Utilities (`cusolver_utils.jl`)

#### DenseLUFactors
Wrapper for cuSOLVER's LU factorization:
- Uses `cusolverDnXgetrf` for factorization
- Uses `cusolverDnXgetrs` for solve
- Supports Float32, Float64, ComplexF32, ComplexF64

#### Key Functions
- `dense_lu!(A)`: In-place LU factorization
- `dense_lu_solve!(factors, b)`: Solve using pre-computed factors

### 3. GPU-Batched Schur Complement (`schur.jl`)

#### build_augmented_system_per_gpu
Builds one large augmented system per GPU combining all scenarios assigned to that GPU:

**Augmented Matrix Structure:**
```
K_aug = [K₀   -B₁ᵀ  -B₂ᵀ  ⋯  ]
        [B₁    K₁    0    ⋯  ]
        [B₂    0     K₂   ⋯  ]
        [⋮     ⋮     ⋮    ⋱  ]
```

**Algorithm:**
1. Compute total size: n_total = n₀ + Σ nᵢ for scenarios on this GPU
2. Allocate sparse matrix in COO format
3. Copy K₀ to top-left block (with zero placeholder for true K₀ diagonal)
4. For each scenario i on this GPU:
   - Copy Kᵢ to diagonal block
   - Copy Bᵢ to off-diagonal blocks (transposed for symmetry)
5. Convert to CSR format for CUDSS

**Key advantage**: One large factorization per GPU instead of many small ones.

#### compute_gpu_schur
Extracts Schur complement from the augmented system using CUDSS:

**Algorithm:**
1. Create CUDSS solver with structure='G' (general) and view='F' (full)
   - **Critical**: Augmented system is NOT SPD even if original system is
2. Set `schur_mode = 1` to enable Schur complement computation
3. Create binary `schur_indices`: first n₀ rows/cols = 1, rest = 0
4. Perform analysis and factorization with `asynchronous=false`
5. Extract Schur complement shape and dense matrix
6. Return solver (for preconditioner), Schur complement, and scenario indices

**Key CUDSS Features Used:**
- `cudss_set("schur_mode", 1)`: Enable Schur complement computation
- `cudss_set("user_schur_indices", indices)`: Mark first-stage variables
- `cudss_get("schur_shape")`: Query Schur complement dimensions
- `cudss_get("schur_matrix")`: Extract dense Schur complement
- **Critical**: `asynchronous=false` for all CUDSS calls to avoid race conditions
#### aggregate_schur_complements
Combines GPU-local Schur complements (not scenario-level):
- C = Σ_gpu S_gpu
- Each S_gpu already includes contribution from K₀
- Returns dense matrix (first-stage is typically small)

### 4. Matrix-Vector Product (`matvec.jl`)

#### matvec!
Computes r = K * z for arrowhead structure:

**Algorithm:**
```
r₀ = K₀ * z₀ + Σᵢ Bᵢᵀ * zᵢ
rᵢ = Kᵢ * zᵢ + Bᵢ * z₀  for each i
```

**Implementation Details:**
- Uses sparse matrix-vector products for Kᵢ
- Uses CUBLAS `gemv` for dense coupling matrices
- Handles multi-GPU by switching device context

### 5. Preconditioner (`preconditioner.jl`)

#### apply_preconditioner!
Applies M⁻¹ using GPU-batched factored augmented systems:

**Algorithm (Block Triangular Solve with Augmented Systems):**
```
1. Forward pass (per GPU, parallel):
   - Build augmented RHS vector: r_aug = [r₀; r_scenarios_on_GPU]
   - Apply CUDSS partial solve: "solve_fwd_schur" with asynchronous=false
   - Extract first n₀ components as r̃_gpu (local contribution to first-stage RHS)

2. Reduce: r̃₀ = Σ_gpu r̃_gpu
   - Aggregate first-stage contributions across all GPUs

3. Dense solve: Δz₀ = C⁻¹ r̃₀
   - Uses cuSOLVER LU factors for global Schur complement

4. Backward pass (per GPU, parallel):
   - Broadcast Δz₀ to all GPUs
   - Build augmented vector: x_aug = [Δz₀; zeros(scenarios)]
   - Apply CUDSS partial solve: "solve_bwd_schur" with asynchronous=false
   - Extract scenario components as Δzᵢ
```

**Key Features:**
- Uses preallocated `workspace.r_aug_gpus` and `workspace.r_tilde_0`
- **Critical**: `asynchronous=false` prevents race conditions in multi-GPU reduction
- CUDSS Partial Solves:
  - `"solve_fwd_schur"`: Combines `SOLVE_FWD_PERM + SOLVE_FWD + SOLVE_DIAG`
  - `"solve_bwd_schur"`: Combines `SOLVE_BWD + SOLVE_BWD_PERM`

### 6. Factorization (`factorize.jl`)

#### LinearAlgebra.factorize
Main factorization routine implementing Julia's standard interface:

**Signature:** `LinearAlgebra.factorize(system::ArrowheadSystem; tol, maxiter, verbose)`

**Algorithm:**
```
1. Create ArrowheadSolver with workspace preallocation

2. For each GPU (parallel across GPUs):
   a. Build augmented system for all scenarios on this GPU
   b. Factor K_aug with CUDSS Schur complement mode (asynchronous=false)
   c. Extract S_gpu = local Schur complement

3. Aggregate: C = Σ_gpu S_gpu

4. Factor dense C using cuSOLVER LU

5. Mark solver as factorized
```

**Multi-GPU Strategy:**
- Each scenario assigned to a GPU via `device_map`
- **GPU-batched**: All scenarios on same GPU factored together in one augmented system
- Only Schur complements transferred between GPUs
- First-stage factorization replicated on all GPUs for preconditioner

**Update Interface:** `LinearAlgebra.factorize(solver, system_new)` allows refactorization with new matrix values.

### 7. BiCGStab Solver (`bicgstab.jl`)

#### bicgstab_solve!
Krylov.jl-based iterative refinement:

**Implementation:**
```julia
1. Create ArrowheadOperator wrapping the system
   - Implements LinearAlgebra.mul! for matrix-vector products
   - Provides size() and eltype() for Krylov.jl

2. Create ArrowheadPreconditioner wrapping the solver
   - Implements LinearAlgebra.ldiv! for preconditioner application

3. Initialize or reuse Krylov.BicgstabWorkspace
   - Lazy allocation on first solve
   - Stores in solver.workspace.krylov_workspace

4. Call Krylov.bicgstab! with workspace and operators
   - Uses preallocated workspace for in-place operations
   - Convergence based on atol/rtol

5. Extract solution and statistics from workspace
```

**Key Features:**
- **Krylov.jl Integration**: Uses professional implementation with extensive testing
- **Right Preconditioning**: Uses `N=M, ldiv=true` for right preconditioning
  - BiCGSTAB monitors true residual `||b - Ax||`, not preconditioned residual
  - Internal Krylov residual matches actual solution residual
  - Better convergence monitoring than left preconditioning for approximate preconditioners
- **Custom Operators**: ArrowheadOperator and ArrowheadPreconditioner
- **Preallocated Workspace**: All BiCGStab vectors in solver.workspace
- **Lazy Krylov Workspace**: Allocated on first solve, reused thereafter
- **Logging**: Uses @info macro, controlled by verbose flag
- **AbstractVector Interface**: ArrowheadVector implements full interface for Krylov.jl

### 8. LinearAlgebra Interface (`solver.jl`)

Implements standard Julia interfaces for solving linear systems:

**Backslash operator:**
```julia
Base.:\(solver::ArrowheadSolver, b) = ...  # Allocating solve
```

**In-place solves:**
```julia
LinearAlgebra.ldiv!(x, solver, b)  # x = solver \ b
LinearAlgebra.ldiv!(solver, b)     # b = solver \ b (overwrites b)
```

**High-level interface:**
```julia
solve!(solver, b)  # Returns (x, stats) with convergence info
```

## Key Implementation Decisions

### 1. GPU-Batched Schur Complement
**Key Innovation**: Instead of one factorization per scenario, we build one augmented system per GPU:
- **Advantage**: Reduces number of CUDSS factorizations from n_scenarios to n_gpus
- **Trade-off**: Larger sparse matrices but fewer factorizations
- **Critical**: Augmented system structure is general ('G'), not SPD, even if original is SPD

### 2. CUDSS Schur Complement Mode
Instead of manually computing -BᵀK⁻¹B, we use CUDSS's built-in Schur complement mode:
- Mark first n₀ rows/columns with `user_schur_indices`
- CUDSS computes Schur complement during factorization
- Handles pivot perturbations consistently
- Stores factors for later use in preconditioner solves

### 3. Explicit Synchronization
**Critical for Correctness**: All CUDSS calls use `asynchronous=false`:
- Default CUDSS behavior is asynchronous (operations queued, not completed)
- Multi-GPU aggregation requires results to be ready
- Without synchronization: race conditions and incorrect results
- **Discovered empirically**: Residuals improved 40x after adding synchronization

### 4. Partial Solves for Preconditioner
CUDSS provides phases for partial triangular solves with Schur complement:
- `solve_fwd_schur`: Forward elimination, produces condensed RHS for first-stage
- `solve_bwd_schur`: Backward substitution, completes solve from first-stage solution
- These are essential for the GPU-batched block triangular solve structure

### 5. Krylov.jl Integration
Uses Krylov.jl instead of custom BiCGStab implementation:
- **Professional implementation** with extensive testing
- **Custom operators**: ArrowheadOperator and ArrowheadPreconditioner wrappers
- **AbstractVector interface**: ArrowheadVector subtypes AbstractVector{T}
- **Workspace management**: Lazy allocation, reused across solves
- **Avoids scalar indexing**: All operations optimized for GPU

### 6. Dense First-Stage Schur Complement
The aggregated Schur complement C is stored dense because:
- First-stage dimension n₀ is typically small (< 1000)
- Dense LU is very fast on GPU for small matrices
- Simplifies the preconditioner solve

### 7. Multi-GPU via Device Switching
Rather than MPI or NCCL, we use:
- CUDA.jl's `device!()` to switch contexts
- Explicit device assignments in `device_map`
- Assumes unified memory or manual transfers
- Simpler for single-node multi-GPU

## Testing

The test suite (`test/runtests.jl`) includes:
1. **Basic construction**: Verify data structure creation
2. **Vector operations**: Test dot, norm, axpy, etc.
3. **Matrix-vector products**: Verify structure is correct
4. **Small system solve**: End-to-end solve with convergence check
5. **Multi-GPU distribution**: Test scenario distribution (if available)

## Performance Considerations

### Bottlenecks
1. **Dense Schur complement factorization**: O(n0³)
2. **Coupling matrix operations**: Dense matrix-vector products
3. **GPU-GPU communication**: Schur complement transfers

### Optimization Opportunities
1. **Overlapping computation**: Use CUDA streams for scenarios
2. **Batch operations**: Batch CUBLAS calls when possible
3. **Memory pooling**: Reuse temporary vectors
4. **Pivot control**: Investigate CUDSS pivot options

## Future Enhancements

1. **NCCL Integration**: For distributed multi-node scaling
2. **Mixed Precision**: Use FP32 for preconditioner, FP64 for residuals
3. **Adaptive BiCGStab**: Adjust tolerance based on progress
4. **Sparse Schur Complement**: For large n0, keep C sparse
5. **Krylov.jl Integration**: Create proper linear operator interface
6. **Benchmark Suite**: Systematic performance testing
7. **Memory Profiling**: Track GPU memory usage

## References

Algorithm based on:
- Paper: [To be added - the 130908737.pdf document]
- CUDSS Documentation: NVIDIA cuDSS Library
- cuSOLVER Documentation: NVIDIA cuSOLVER Library
- Krylov.jl: Julia package for iterative solvers

## Author

Michel Schanen (mschanen@anl.gov)
Argonne National Laboratory

## Date

November 2025

