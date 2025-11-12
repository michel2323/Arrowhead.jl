# Schur complement computation using CUDSS

"""
    build_augmented_system_per_gpu(
        system::ArrowheadSystem{T,INT},
        device_id::Int
    ) -> (K_aug::CuSparseMatrixCSR{T,INT}, scenario_indices::Vector{Int}, n_total::Int)

Build an augmented sparse matrix combining all scenarios on a specific GPU.

The augmented matrix has the structure:
```
K_aug = [0      B₁ᵀ  B₂ᵀ  ⋯  Bₖᵀ]
        [B₁     K₁   0   ⋯  0  ]
        [B₂     0    K₂  ⋯  0  ]
        [⋮      ⋮    ⋮   ⋱  ⋮  ]
        [Bₖ     0    0   ⋯  Kₖ ]
```

where k is the number of scenarios on this GPU.

# Arguments
- `system`: ArrowheadSystem containing all scenario data
- `device_id`: GPU device index (0-based)

# Returns
- `K_aug`: Augmented sparse matrix on the specified GPU
- `scenario_indices`: List of scenario indices on this GPU
- `n_total`: Total dimension of augmented system
"""
function build_augmented_system_per_gpu(
    system::ArrowheadSystem{T,INT},
    device_id::Int
) where {T,INT}

    # Switch to the appropriate device
    CUDA.device!(device_id)

    # Find scenarios on this GPU
    scenario_indices = findall(x -> x == device_id, system.device_map)
    n_scenarios_gpu = length(scenario_indices)

    if n_scenarios_gpu == 0
        error("No scenarios assigned to GPU $device_id")
    end

    n0 = system.n0

    # Compute total dimension
    n_total = n0
    for idx in scenario_indices
        n_total += system.scenario_dims[idx]
    end

    # Build the augmented matrix in COO format, then convert to CSR
    rows = Vector{INT}()
    cols = Vector{INT}()
    vals = Vector{T}()

    # Current row/col offset (start after the n0×n0 zero block)
    offset = n0

    # Add each scenario block and its coupling
    for idx in scenario_indices
        K_scen = system.K_scenarios[idx]
        B_coup = system.B_coupling[idx]
        n_scen = system.scenario_dims[idx]

        # Get K_scenario data on CPU for processing
        K_cpu = SparseMatrixCSC(K_scen)
        K_rows, K_cols, K_vals = findnz(K_cpu)

        # Add K_scenario block to diagonal
        for i in 1:length(K_vals)
            push!(rows, INT(K_rows[i] + offset))
            push!(cols, INT(K_cols[i] + offset))
            push!(vals, K_vals[i])
        end

        # Add B coupling blocks
        # B is n_scen × n0, we need to add it in two places:
        # 1. B in lower-left: rows [offset:offset+n_scen-1], cols [1:n0]
        # 2. Bᵀ in upper-right: rows [1:n0], cols [offset:offset+n_scen-1]
        B_cpu = Array(B_coup)

        for j in 1:n0
            for i in 1:n_scen
                val = B_cpu[i, j]
                if abs(val) > eps(real(T))  # Only add non-zeros
                    # B block (lower-left)
                    push!(rows, INT(i + offset))
                    push!(cols, INT(j))
                    push!(vals, val)

                    # Bᵀ block (upper-right)
                    push!(rows, INT(j))
                    push!(cols, INT(i + offset))
                    push!(vals, val)
                end
            end
        end

        offset += n_scen
    end

    # Create sparse matrix from COO format
    A_cpu = sparse(rows, cols, vals, n_total, n_total)

    # Convert to GPU CSR format
    K_aug = CuSparseMatrixCSR(A_cpu)

    return K_aug, scenario_indices, n_total
end

"""
    compute_gpu_schur(
        system::ArrowheadSystem{T,INT},
        device_id::Int;
        verbose::Bool = false
    ) -> (solver, S_gpu::CuMatrix{T}, scenario_indices::Vector{Int})

Compute aggregated Schur complement for all scenarios on a specific GPU using CUDSS.

This function builds an augmented system containing all scenarios on the GPU,
then uses CUDSS's Schur complement mode to compute:
S_gpu = -Σᵢ(Bᵢᵀ Kᵢ⁻¹ Bᵢ) for all scenarios i on this GPU

# Arguments
- `system`: ArrowheadSystem containing all scenario data
- `device_id`: GPU device index (0-based)

# Returns
- `solver`: CUDSS solver object for the augmented system
- `S_gpu`: Dense Schur complement matrix (n0 × n0) for this GPU
- `scenario_indices`: List of scenario indices on this GPU
"""
function compute_gpu_schur(
    system::ArrowheadSystem{T,INT},
    device_id::Int;
    verbose::Bool = false
) where {T,INT}

    # Switch to the appropriate device
    CUDA.device!(device_id)

    # Build augmented system for this GPU
    K_aug, scenario_indices, n_total = build_augmented_system_per_gpu(system, device_id)

    n0 = system.n0

    verbose && @info "Built augmented system" device_id size=(n_total, n_total) n_scenarios=length(scenario_indices)

    # Create solver with Schur complement mode
    # Note: Use "G" (general) structure even if original system is SPD,
    # because the augmented matrix has zeros in top-left block
    aug_structure = "G"  # Always use general structure for augmented system
    aug_view = 'F'       # Full matrix view
    solver = CUDSS.CudssSolver(K_aug, aug_structure, aug_view)
    CUDSS.cudss_set(solver, "schur_mode", 1)

    # Set Schur indices: 1 for first n0 (Schur complement), 0 for rest (to be factored out)
    schur_indices = Cint[i <= n0 ? 1 : 0 for i in 1:n_total]
    CUDSS.cudss_set(solver, "user_schur_indices", schur_indices)

    # Factorize (computes Schur complement internally)
    x_dummy = CUDA.zeros(T, n_total)
    b_dummy = CUDA.zeros(T, n_total)

    CUDSS.cudss("analysis", solver, x_dummy, b_dummy; asynchronous=false)
    CUDSS.cudss("factorization", solver, x_dummy, b_dummy; asynchronous=false)

    # Extract Schur complement shape
    (nrows_S, ncols_S, nnz_S) = CUDSS.cudss_get(solver, "schur_shape")

    verbose && @info "Extracted Schur complement" device_id shape=(nrows_S, ncols_S)

    # Extract as dense matrix
    S_gpu = CuMatrix{T}(undef, nrows_S, ncols_S)
    cudss_matrix = CUDSS.CudssMatrix(S_gpu)
    CUDSS.cudss_set(solver, "schur_matrix", cudss_matrix.matrix)
    CUDSS.cudss_get(solver, "schur_matrix")

    # Return solver (for later use in preconditioner) and Schur complement
    return solver, S_gpu, scenario_indices
end

"""
    aggregate_schur_complements(
        schur_complements::Vector{CuMatrix{T}},
        K0::CuSparseMatrixCSR{T,INT}
    ) -> CuMatrix{T}

Aggregate scenario Schur complements with the first-stage matrix.

Computes: C = K₀ + Σᵢ Sᵢ

# Arguments
- `schur_complements`: Vector of Schur complement matrices from each scenario
- `K0`: First-stage sparse matrix

# Returns
- Dense matrix C = K₀ + Σᵢ Sᵢ
"""
function aggregate_schur_complements(
    schur_complements::Vector{CuMatrix{T}},
    K0::CuSparseMatrixCSR{T,INT}
) where {T,INT}

    n0 = size(K0, 1)

    # Convert K0 to dense (assuming it's small)
    K0_dense = CuMatrix{T}(K0)

    # Start with K0
    C = copy(K0_dense)

    # Add all Schur complements
    for S in schur_complements
        C .+= S
    end

    return C
end

