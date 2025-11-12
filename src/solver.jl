# LinearAlgebra interface for ArrowheadSolver

"""
    LinearAlgebra.ldiv!(x::ArrowheadVector{T}, solver::ArrowheadSolver{T,INT,BCT}, b::ArrowheadVector{T})

Solve `x = solver \\ b` in-place (similar to `ldiv!(x, F, b)` for LU factorizations).

Uses BiCGStab with the factorized system as a preconditioner.

# Arguments
- `x`: Solution vector (will be overwritten)
- `solver`: Factorized ArrowheadSolver
- `b`: Right-hand side vector

# Returns
- `x`: Solution vector
- `stats`: Convergence statistics

# Example
```julia
solver = factorize(system)
x = ArrowheadVector(system, Val{:zeros})
ldiv!(x, solver, b)
```
"""
function LinearAlgebra.ldiv!(
    x::ArrowheadVector{T},
    solver::ArrowheadSolver{T,INT,BCT},
    b::ArrowheadVector{T}
) where {T,INT,BCT}

    @assert solver.factorized "Solver must be factorized before solving"

    # Use BiCGStab with workspace
    _, stats = bicgstab_solve!(solver, b, x)

    return x
end

"""
    LinearAlgebra.ldiv!(solver::ArrowheadSolver{T,INT,BCT}, b::ArrowheadVector{T})

Solve in-place, overwriting `b` with the solution (similar to `ldiv!(F, b)` for LU).

Uses BiCGStab with the factorized system as a preconditioner.

# Arguments
- `solver`: Factorized ArrowheadSolver
- `b`: Right-hand side vector (will be overwritten with solution)

# Returns
- `b`: Solution vector (modified in-place)

# Example
```julia
solver = factorize(system)
ldiv!(solver, b)  # b now contains solution
```
"""
function LinearAlgebra.ldiv!(
    solver::ArrowheadSolver{T,INT,BCT},
    b::ArrowheadVector{T}
) where {T,INT,BCT}

    @assert solver.factorized "Solver must be factorized before solving"

    # Use temporary workspace vector
    x = solver.workspace.x
    zero!(x)

    # Solve into x
    bicgstab_solve!(solver, b, x)

    # Copy result back to b
    copy!(b, x)

    return b
end

"""
    Base.:\\(solver::ArrowheadSolver{T,INT,BCT}, b::ArrowheadVector{T})

Solve `solver \\ b` with allocation (similar to `F \\ b` for LU factorizations).

Uses BiCGStab with the factorized system as a preconditioner.

# Arguments
- `solver`: Factorized ArrowheadSolver
- `b`: Right-hand side vector

# Returns
- `x`: Solution vector (newly allocated)
- `stats`: Convergence statistics (as second return value)

# Example
```julia
solver = factorize(system)
x, stats = solve!(solver, b)
```
"""
function Base.:\(
    solver::ArrowheadSolver{T,INT,BCT},
    b::ArrowheadVector{T}
) where {T,INT,BCT}

    @assert solver.factorized "Solver must be factorized before solving"

    # Allocate solution vector
    x = ArrowheadVector(solver.system, Val{:zeros})

    # Solve
    _, stats = bicgstab_solve!(solver, b, x)

    return x, stats
end

"""
    solve!(solver::ArrowheadSolver{T,INT,BCT}, b::ArrowheadVector{T}; tol=nothing, maxiter=nothing)

High-level solve interface that returns solution and convergence statistics.

Allows overriding solver tolerances at solve time.

# Arguments
- `solver`: Factorized ArrowheadSolver
- `b`: Right-hand side vector

# Keyword Arguments
- `tol`: Convergence tolerance (overrides solver.tol if provided). Used for both atol and rtol
- `maxiter`: Maximum iterations (overrides solver.maxiter if provided)

# Returns
- `x`: Solution vector (newly allocated)
- `stats`: Convergence statistics with fields `niter`, `residual`, `converged`

# Examples
```julia
# Use solver defaults (set during factorization)
solver = factorize(system; tol=1e-6, maxiter=100)
x, stats = solve!(solver, b)

# Override tolerance for this solve
x, stats = solve!(solver, b; tol=1e-8, maxiter=200)
```
"""
function solve!(
    solver::ArrowheadSolver{T,INT,BCT},
    b::ArrowheadVector{T};
    tol::Union{Nothing,Float64} = nothing,
    maxiter::Union{Nothing,Int} = nothing
) where {T,INT,BCT}

    @assert solver.factorized "Solver must be factorized before solving"

    # Allocate solution vector
    x = ArrowheadVector(solver.system, Val{:zeros})

    # Solve with optional tolerance overrides
    _, stats = bicgstab_solve!(solver, b, x; tol=tol, maxiter=maxiter)

    return x, stats
end

