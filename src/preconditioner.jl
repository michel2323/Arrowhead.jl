# Preconditioner solve using factorized arrowhead system

"""
    apply_preconditioner!(
        Δz::ArrowheadVector{T},
        solver::ArrowheadSolver{T,INT,BCT},
        r::ArrowheadVector{T}
    )

Apply the preconditioner: Δz = K̃⁻¹ r using stored factorization and workspace.

This implements the block triangular solve using augmented GPU systems:
1. Forward elimination: Partial forward solve on each GPU's augmented system
2. Reduce: r̃₀ = r₀ - contribution from all GPUs
3. Solve first stage: Δz₀ = C̃⁻¹ r̃₀
4. Back-substitution: Partial backward solve on each GPU's augmented system

Uses preallocated workspace from solver to avoid allocations.

Note: Due to pivot perturbations in CUDSS, K̃ ≈ K, so this is an approximate solve.

# Arguments
- `Δz`: Output solution vector (will be overwritten)
- `solver`: Factorized ArrowheadSolver
- `r`: Right-hand side vector
"""
function apply_preconditioner!(
    Δz::ArrowheadVector{T},
    solver::ArrowheadSolver{T,INT,BCT},
    r::ArrowheadVector{T}
) where {T,INT,BCT}

    @assert solver.factorized "Solver must be factorized before applying preconditioner"

    system = solver.system
    n0 = system.n0

    # Use preallocated workspace
    r_aug_gpus = solver.workspace.r_aug_gpus
    r_tilde_0 = solver.workspace.r_tilde_0

    # Step 1: Forward elimination using augmented solvers on each GPU
    # Compute partial forward solve up to Schur complement
    for (device_id, gpu_solver) in solver.gpu_solvers
        CUDA.device!(device_id)

        # Build augmented RHS vector for this GPU (or reuse if already allocated)
        if !haskey(r_aug_gpus, device_id)
            # Compute size for this GPU's augmented vector
            n_total = n0
            for idx in solver.gpu_to_scenarios[device_id]
                n_total += system.scenario_dims[idx]
            end
            r_aug_gpus[device_id] = CUDA.zeros(T, n_total)
        end

        r_aug = r_aug_gpus[device_id]

        # Pack r into augmented format
        r_aug[1:n0] .= r.z0
        offset = n0
        for idx in solver.gpu_to_scenarios[device_id]
            n_scen = system.scenario_dims[idx]
            r_aug[offset+1:offset+n_scen] .= r.z_scenarios[idx]
            offset += n_scen
        end

        # Create CUDSS matrix views
        b_matrix = CUDSS.CudssMatrix(r_aug)
        x_matrix = CUDSS.CudssMatrix(r_aug)  # overwrites in place

        # Partial forward solve: computes condensed RHS for Schur complement
        # After this, the first n0 components contain r̃_gpu = effect of this GPU's scenarios
        CUDSS.cudss("solve_fwd_schur", gpu_solver, x_matrix, b_matrix; asynchronous=false)
    end

    # Step 2: Reduce to compute RHS for first-stage system
    # The first n0 components of each x_aug contain the condensed RHS
    # r̃₀ = sum over GPUs of (first n0 components)
    CUDA.device!(0)
    fill!(r_tilde_0, zero(T))  # Zero out workspace

    for (device_id, x_aug) in r_aug_gpus
        CUDA.device!(device_id)
        # Add contribution from this GPU
        r_tilde_0 .+= @view(x_aug[1:n0])
    end

    # Step 3: Solve first-stage dense system: C̃ Δz₀ = r̃₀
    CUDA.device!(0)
    copyto!(Δz.z0, r_tilde_0)
    dense_lu_solve!(solver.C_factors, Δz.z0)

    # Step 4: Back-substitution using augmented solvers
    for (device_id, gpu_solver) in solver.gpu_solvers
        CUDA.device!(device_id)

        # Get the forward solve result for this GPU
        x_aug = r_aug_gpus[device_id]

        # Inject Δz₀ into the first n0 components
        # (These are the solution for the Schur complement system)
        x_aug[1:n0] .= Δz.z0

        # Create CUDSS matrix views
        # Note: For backward solve, solution is in b_matrix position, RHS in x_matrix
        b_matrix = CUDSS.CudssMatrix(x_aug)
        x_matrix = CUDSS.CudssMatrix(x_aug)

        # Partial backward solve: completes the full solve using the Schur complement solution
        CUDSS.cudss("solve_bwd_schur", gpu_solver, b_matrix, x_matrix; asynchronous=false)

        # Extract scenario solutions from augmented vector
        offset = n0
        for idx in solver.gpu_to_scenarios[device_id]
            n_scen = system.scenario_dims[idx]
            Δz.z_scenarios[idx] .= @view(x_aug[offset+1:offset+n_scen])
            offset += n_scen
        end
    end

    return Δz
end

# Backward compatibility alias
const solve_preconditioner! = apply_preconditioner!

"""
    solve_preconditioner(
        factors::ArrowheadFactors{T,INT},
        r::ArrowheadVector{T}
    ) -> ArrowheadVector{T}

Non-mutating version of preconditioner solve.

Allocates a new ArrowheadVector for the result.
"""
function solve_preconditioner(
    factors::ArrowheadFactors{T,INT},
    r::ArrowheadVector{T}
) where {T,INT}
    Δz = ArrowheadVector(factors.system, Val{:zeros})
    solve_preconditioner!(Δz, factors, r)
    return Δz
end

