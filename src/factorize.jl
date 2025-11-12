# Factorization of arrowhead systems

"""
    LinearAlgebra.factorize(solver::ArrowheadSolver{T,INT,BCT})

Factorize an arrowhead system in-place (similar to `lu!(F, A)`).

This performs the following steps:
1. For each GPU (distributed computation):
   - Build augmented system with all scenarios on that GPU
   - Factor using CUDSS Schur complement mode
   - Extract Schur complement S_gpu = -Σᵢ(Bᵢᵀ K̃ᵢ⁻¹ Bᵢ) for scenarios on GPU
2. Aggregate Schur complements: C = K₀ + Σ_gpus S_gpu
3. Factor dense C using cuSOLVER LU factorization

# Arguments
- `solver`: ArrowheadSolver structure (will be populated with factorization data)

# Notes
Due to pivot perturbations in CUDSS, the computed factors correspond to K̃ ≈ K,
hence the preconditioner is approximate. BiCGStab iteration corrects for this.
"""
function LinearAlgebra.factorize(solver::ArrowheadSolver{T,INT,BCT}) where {T,INT,BCT}

    system = solver.system
    n_scenarios = system.n_scenarios
    n_gpus = length(solver.gpu_to_scenarios)

    solver.verbose && @info "Factorizing arrowhead system" n_scenarios=n_scenarios n_gpus=n_gpus

    # Step 1: Compute augmented Schur complements per GPU
    solver.verbose && @info "Computing GPU-batched factorizations and Schur complements"

    gpu_idx = 1
    for (device_id, scenario_indices) in sort(collect(solver.gpu_to_scenarios))
        # Compute Schur complement for all scenarios on this GPU
        gpu_solver, S_gpu, _ = compute_gpu_schur(system, device_id; verbose=solver.verbose)

        # Store the solver and Schur complement
        solver.gpu_solvers[device_id] = gpu_solver
        solver.schur_complements[gpu_idx] = S_gpu

        solver.verbose && @info "GPU Schur complement computed" device_id=device_id scenarios=scenario_indices

        gpu_idx += 1
    end

    # Step 2: Aggregate Schur complements
    solver.verbose && @info "Aggregating Schur complements"

    # Switch to device 0 for aggregation
    CUDA.device!(0)

    C = aggregate_schur_complements(solver.schur_complements, system.K0)
    solver.C_aggregated = C

    solver.verbose && @info "Aggregated C matrix" size=size(C)

    # Step 3: Factor dense C using cuSOLVER
    solver.verbose && @info "Factoring dense aggregated Schur complement"

    # Make a copy since dense_lu! modifies in place
    C_copy = copy(C)
    lu_factors = dense_lu!(C_copy)
    solver.C_factors = lu_factors

    # Mark as factorized
    solver.factorized = true

    solver.verbose && @info "Factorization complete"

    return solver
end

"""
    LinearAlgebra.factorize(solver::ArrowheadSolver{T,INT,BCT}, system::ArrowheadSystem{T,INT,BCT})

Update solver with new system and refactorize (similar to `lu!(F, A)`).

# Arguments
- `solver`: ArrowheadSolver to update
- `system`: New ArrowheadSystem

# Returns
- Updated and refactorized solver
"""
function LinearAlgebra.factorize(
    solver::ArrowheadSolver{T,INT,BCT},
    system::ArrowheadSystem{T,INT,BCT}
) where {T,INT,BCT}
    # Update system
    solver.system = system
    solver.factorized = false

    # Rebuild GPU mappings if device map changed
    solver.scenario_to_gpu = Dict{Int,Int}()
    solver.gpu_to_scenarios = Dict{Int,Vector{Int}}()

    for i in 1:system.n_scenarios
        gpu_id = system.device_map[i]
        solver.scenario_to_gpu[i] = gpu_id

        if !haskey(solver.gpu_to_scenarios, gpu_id)
            solver.gpu_to_scenarios[gpu_id] = Int[]
        end
        push!(solver.gpu_to_scenarios[gpu_id], i)
    end

    # Factorize with new system
    return factorize(solver)
end

"""
    LinearAlgebra.factorize(system::ArrowheadSystem{T,INT,BCT}; kwargs...) -> ArrowheadSolver{T,INT,BCT}

Factorize an arrowhead system (similar to `lu(A)` or `cholesky(A)`).

Creates an ArrowheadSolver, allocates workspace, and performs factorization.

# Arguments
- `system`: ArrowheadSystem to factorize

# Keyword Arguments
- `tol::Float64 = 1e-6`: BiCGStab tolerance
- `maxiter::Int = 100`: Maximum BiCGStab iterations
- `verbose::Bool = false`: Print factorization progress

# Returns
- `solver`: Factorized ArrowheadSolver ready for solving

# Example
```julia
system = ArrowheadSystem(K0, K_scenarios, B_coupling, device_map)
solver = factorize(system)
x = solver \\ b
```
"""
function LinearAlgebra.factorize(
    system::ArrowheadSystem{T,INT,BCT};
    tol::Float64 = 1e-6,
    maxiter::Int = 100,
    verbose::Bool = false
) where {T,INT,BCT}
    solver = ArrowheadSolver(system; tol=tol, maxiter=maxiter, verbose=verbose)
    factorize(solver)
    return solver
end

