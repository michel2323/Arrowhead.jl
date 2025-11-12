# BiCGStab solver using Krylov.jl

using Krylov
using LinearAlgebra

"""
    ArrowheadOperator{T,INT}

Linear operator wrapper for use with Krylov.jl solvers.

This allows Krylov.jl to apply the arrowhead matrix K to vectors
without explicitly forming the full matrix.
"""
struct ArrowheadOperator{T,INT}
    system::ArrowheadSystem{T,INT}
    temp::ArrowheadVector{T}  # Temporary storage for matrix-vector products
end

function ArrowheadOperator(system::ArrowheadSystem{T,INT}) where {T,INT}
    temp = ArrowheadVector(system, Val{:zeros})
    ArrowheadOperator{T,INT}(system, temp)
end

# Implement matrix-vector product for Krylov.jl
function LinearAlgebra.mul!(y::ArrowheadVector{T}, A::ArrowheadOperator{T,INT}, x::ArrowheadVector{T}) where {T,INT}
    matvec!(y, A.system, x)
    return y
end

# Required for Krylov.jl
function Base.size(A::ArrowheadOperator)
    system = A.system
    n_total = system.n0 + sum(system.scenario_dims)
    return (n_total, n_total)
end

Base.eltype(::ArrowheadOperator{T,INT}) where {T,INT} = T

"""
    ArrowheadPreconditioner{T,INT,BCT}

Preconditioner wrapper for use with Krylov.jl solvers.

This allows Krylov.jl to apply the preconditioner M⁻¹ to vectors.
"""
struct ArrowheadPreconditioner{T,INT,BCT}
    solver::ArrowheadSolver{T,INT,BCT}
end

# Implement preconditioner application for Krylov.jl
function LinearAlgebra.ldiv!(y::ArrowheadVector{T}, P::ArrowheadPreconditioner{T,INT,BCT}, x::ArrowheadVector{T}) where {T,INT,BCT}
    apply_preconditioner!(y, P.solver, x)
    return y
end

"""
    bicgstab_solve!(
        solver::ArrowheadSolver{T,INT,BCT},
        b::ArrowheadVector{T},
        x::ArrowheadVector{T} = solver.workspace.x;
        tol::Union{Nothing,Float64} = nothing,
        maxiter::Union{Nothing,Int} = nothing
    )

Solve the arrowhead system K x = b using Krylov.jl's BiCGStab with preconditioning.

Uses Krylov.jl's optimized BiCGStab implementation with custom ArrowheadVector
workspace and the factorized system as a preconditioner.

# Arguments
- `solver`: Factorized ArrowheadSolver (contains workspace and parameters)
- `b`: Right-hand side vector
- `x`: Solution vector (will be overwritten; defaults to solver workspace)

# Keyword Arguments
- `tol`: Convergence tolerance (defaults to solver.tol). Used for both atol and rtol
- `maxiter`: Maximum iterations (defaults to solver.maxiter)

# Returns
- `x`: Solution vector (modified in-place)
- `stats`: Named tuple with convergence information (niter, residual, converged)
"""
function bicgstab_solve!(
    solver::ArrowheadSolver{T,INT,BCT},
    b::ArrowheadVector{T},
    x::ArrowheadVector{T} = solver.workspace.x;
    tol::Union{Nothing,Float64} = nothing,
    maxiter::Union{Nothing,Int} = nothing
) where {T,INT,BCT}

    @assert solver.factorized "Solver must be factorized before solving"

    system = solver.system

    # Use provided tolerances or fall back to solver defaults
    atol = rtol = isnothing(tol) ? solver.tol : tol
    itmax = isnothing(maxiter) ? solver.maxiter : maxiter

    solver.verbose && @info "Starting BiCGStab solver" tolerance=atol maxiter=itmax

    # Create operator and preconditioner for Krylov.jl
    A = ArrowheadOperator(system)
    M = ArrowheadPreconditioner(solver)

    # Initialize Krylov workspace if not already done
    if solver.workspace.krylov_workspace === nothing
        kc = Krylov.KrylovConstructor(b)
        solver.workspace.krylov_workspace = Krylov.BicgstabWorkspace(kc)
    end
    workspace = solver.workspace.krylov_workspace

    # Solve with Krylov.jl BiCGStab
    # verbose: 0=quiet, 1=iterations+final, 2=iterations+residuals
    # Note: atol and rtol are set to the same value for consistency
    Krylov.bicgstab!(
        workspace, A, b;
        M=M, ldiv=true,
        atol=atol, rtol=rtol,
        itmax=itmax,
        verbose=solver.verbose ? 2 : 0
    )

    # Extract solution
    x_krylov = Krylov.solution(workspace)
    copy!(x, x_krylov)

    # Get convergence info from workspace
    stats_krylov = workspace.stats
    niter = stats_krylov.niter
    solved = stats_krylov.solved

    # Compute final residual for stats
    temp = solver.workspace.temp
    matvec!(temp, system, x)
    axpy!(-one(T), b, temp)  # temp = Ax - b
    residual = norm(temp) / norm(b)

    solver.verbose && @info "BiCGStab completed" iterations=niter converged=solved residual=residual

    return x, (niter=niter, residual=residual, converged=solved)
end
