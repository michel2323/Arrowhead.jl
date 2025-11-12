# Arrowhead.jl

A Julia package for solving large-scale arrowhead-structured linear systems using GPU-batched Schur complement methods on multiple GPUs with NVIDIA's CUDSS library and Krylov.jl iterative solvers.

## Overview

Arrowhead.jl implements an efficient GPU-batched augmented factorization approach with BiCGStab iterative refinement for solving linear systems with arrowhead structure:

```
K = [K₀    B₁ᵀ  B₂ᵀ  ⋯  Bₙᵀ]
    [B₁    K₁   0   ⋯  0  ]
    [B₂    0    K₂  ⋯  0  ]
    [⋮     ⋮    ⋮   ⋱  ⋮  ]
    [Bₙ    0    0   ⋯  Kₙ ]
```

This structure commonly arises in:
- **Stochastic optimization** (scenario-based problems)
- **Domain decomposition methods**
- **Multistage optimization**
- **Network optimization with central hub**

## Key Features

- **Multi-GPU Support**: Distribute scenarios across multiple GPUs for parallel computation
- **GPU-Batched Schur Complement**: One augmented system per GPU using CUDSS's Schur complement mode
- **Krylov.jl Integration**: BiCGStab solver with custom operators and preconditioners
- **Asynchronous GPU Operations**: Explicit synchronization for correct multi-GPU results
- **LinearAlgebra API**: Standard `factorize`, `\`, and `ldiv!` interfaces
- **Flexible Structure Support**: Handles general, symmetric, Hermitian, and positive-definite matrices

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/michel2323/Arrowhead.jl")
```

**Requirements:**
- Julia ≥ 1.10
- CUDA-capable GPU
- CUDA.jl ≥ 5.0
- CUDSS.jl ≥ 0.6
- Krylov.jl ≥ 0.10

## Quick Start

```julia
using Arrowhead
using CUDA
using LinearAlgebra
using SparseArrays

# Problem dimensions
n0 = 10              # First-stage dimension
n_scenarios = 8      # Number of scenarios
n_scenario = 50      # Dimension of each scenario

# Create first-stage matrix (SPD)
A0 = sprand(Float64, n0, n0, 0.8)
A0 = A0 + A0' + 5.0 * I
K0 = CuSparseMatrixCSR(A0)

# Create scenario matrices and coupling
K_scenarios = []
B_coupling = []

for i in 1:n_scenarios
    # Scenario matrix
    A_scen = sprand(Float64, n_scenario, n_scenario, 0.5)
    A_scen = A_scen + A_scen' + 3.0 * I
    push!(K_scenarios, CuSparseMatrixCSR(A_scen))

    # Coupling matrix
    B = CUDA.randn(Float64, n_scenario, n0) * 0.1
    push!(B_coupling, B)
end

# Create arrowhead system
system = ArrowheadSystem(K0, K_scenarios, B_coupling; structure="SPD", view='U')

# Create right-hand side
b = ArrowheadVector(system, Val{:rand})

# Factorize and solve using LinearAlgebra interface
solver = factorize(system; tol=1e-8, maxiter=100)
x = solver \ b

# Or use the solve! interface
x, stats = solve!(solver, b)
println("Converged in $(stats.niter) iterations, residual = $(stats.residual)")
```

## Multi-GPU Usage

To distribute scenarios across multiple GPUs:

```julia
# Create device mapping (0-based GPU indices)
device_map = [i % CUDA.ndevices() for i in 0:n_scenarios-1]

# Create system with explicit device mapping
system = ArrowheadSystem(K0, K_scenarios, B_coupling, device_map;
                        structure="SPD", view='U')
```

## Algorithm

The solver implements an efficient GPU-batched algorithm with three main components:

### 1. GPU-Batched Factorization (`LinearAlgebra.factorize`)

**Per GPU** (parallel across GPUs):
- Build augmented system combining all scenarios assigned to this GPU:
  ```
  K_aug = [K₀  -B₁ᵀ  -B₂ᵀ  ⋯]
          [B₁   K₁    0    ⋯]
          [B₂   0     K₂   ⋯]
          [⋮    ⋮     ⋮    ⋱]
  ```
- Factor K_aug using CUDSS with Schur complement mode (marks first n₀ rows/cols)
- Extract GPU-local Schur complement S_gpu = K₀ + Σᵢ∈GPU (-Bᵢᵀ Kᵢ⁻¹ Bᵢ)

**Aggregate and factor** the global Schur complement:
- C = Σ_gpus S_gpu
- Factor C using cuSOLVER (LU decomposition)

**Key advantage**: One CUDSS factorization per GPU instead of one per scenario.

### 2. Preconditioner Solve (`apply_preconditioner!`)

Block triangular solve using partial solves with augmented systems:
1. **Forward pass** (per GPU): Partial forward solve on augmented system
2. **Reduce**: Aggregate first-stage contributions across GPUs
3. **Dense solve**: Δz₀ = C⁻¹ r̃₀
4. **Backward pass** (per GPU): Partial backward solve to complete scenarios

**Key feature**: Uses CUDSS's `solve_fwd_schur` and `solve_bwd_schur` phases.

### 3. Krylov.jl BiCGStab Iteration

Iterative refinement to correct for pivot perturbations:
- Custom `ArrowheadOperator` implementing matrix-vector products
- Custom `ArrowheadPreconditioner` using the factorization
- **Right preconditioning**: BiCGSTAB monitors true residual `||b - Ax||` (not preconditioned)
- Preallocated workspace for in-place operations
- Convergence based on relative residual

**Critical**: All CUDSS calls use `asynchronous=false` for correct multi-GPU synchronization.

## API Reference

### Main Types

- **`ArrowheadSystem{T,INT,BCT}`**: Container for the arrowhead-structured system
  - Stores K₀, {Kᵢ}, {Bᵢ}, device mapping, and matrix properties

- **`ArrowheadSolver{T,INT,BCT} <: Factorization{T}`**: Solver with factorization data
  - Contains GPU solvers, Schur complements, workspace, and solver parameters
  - Implements LinearAlgebra's `Factorization` interface

- **`ArrowheadVector{T} <: AbstractVector{T}`**: Distributed vector for arrowhead systems
  - Stores z₀ and {zᵢ} with device mapping
  - Implements full AbstractVector interface for Krylov.jl compatibility

- **`ArrowheadWorkspace{T}`**: Preallocated workspace for in-place operations
  - Stores temporary vectors for BiCGStab and preconditioner
  - Includes lazy-allocated Krylov.jl workspace

### Main Functions

**Factorization:**
- **`LinearAlgebra.factorize(system; tol, maxiter, verbose)`**: Create and factorize solver
- **`LinearAlgebra.factorize(solver, system_new)`**: Update and refactorize

**Solving (LinearAlgebra Interface):**
- **`solver \ b`**: Solve Kx = b (allocating)
- **`LinearAlgebra.ldiv!(x, solver, b)`**: In-place solve: x = K⁻¹b
- **`LinearAlgebra.ldiv!(solver, b)`**: In-place solve: b = K⁻¹b (overwrites b)

**High-Level Interface:**
- **`solve!(solver, b)`**: Solve and return (x, stats)

**Utility Functions:**
- **`matvec!(y, system, x)`**: Matrix-vector product y = Kx
- **`LinearAlgebra.mul!(y, system, x)`**: Alternative matrix-vector product interface

## Performance Tips

1. **GPU Selection**: Distribute scenarios to balance load across GPUs
2. **First-Stage Size**: Keep n0 small (< 1000) for efficient dense solves
3. **Scenario Size**: Sparse scenarios can be large (10K-100K variables)
4. **Coupling Strength**: Weak coupling improves preconditioner quality
5. **BiCGStab Tolerance**: Use 1e-6 to 1e-8 for practical problems

## Examples

See the `examples/` directory for complete examples:
- `simple_arrowhead.jl`: Basic usage with small problem
- More examples coming soon!

## Testing

Run the test suite:

```julia
using Pkg
Pkg.test("Arrowhead")
```

## References

This implementation is based on the algorithm described in the paper (reference to be added).

## License

MIT License (or specify your license)

## Contributing

Contributions are welcome! Please open an issue or pull request.

## Authors

- Michel Schanen (mschanen@anl.gov)

