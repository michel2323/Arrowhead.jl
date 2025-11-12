# Matrix-vector product for arrowhead systems

"""
    matvec!(r::ArrowheadVector{T}, system::ArrowheadSystem{T,INT}, z::ArrowheadVector{T})

Compute the matrix-vector product r = K * z for an arrowhead system.

The arrowhead system has the structure:
```
K = [K₀    B₁ᵀ  B₂ᵀ  ⋯  Bₙᵀ]
    [B₁    K₁   0   ⋯  0  ]
    [B₂    0    K₂  ⋯  0  ]
    [⋮     ⋮    ⋮   ⋱  ⋮  ]
    [Bₙ    0    0   ⋯  Kₙ ]
```

The computation is:
- r₀ = K₀ * z₀ + Σᵢ Bᵢᵀ * zᵢ
- rᵢ = Kᵢ * zᵢ + Bᵢ * z₀  for each scenario i

# Arguments
- `r`: Output vector (will be overwritten)
- `system`: Arrowhead system structure
- `z`: Input vector
"""
function matvec!(
    r::ArrowheadVector{T},
    system::ArrowheadSystem{T,INT},
    z::ArrowheadVector{T}
) where {T,INT}
    
    # Compute r₀ = K₀ * z₀
    # Use sparse matrix-vector product
    mul!(r.z0, system.K0, z.z0)
    
    # Add contributions from scenarios: r₀ += Σᵢ Bᵢᵀ * zᵢ
    # We need to accumulate from all GPUs
    temp_contributions = CUDA.zeros(T, system.n0)
    
    for i in 1:system.n_scenarios
        device_id = system.device_map[i]
        CUDA.device!(device_id)
        
        # Compute Bᵢᵀ * zᵢ and add to temp
        # B is (n_i × n0), z_i is (n_i,), so Bᵀ * z_i is (n0,)
        CUDA.CUBLAS.gemv!('T', one(T), system.B_coupling[i], z.z_scenarios[i], one(T), temp_contributions)
    end
    
    # Add accumulated contributions to r₀
    r.z0 .+= temp_contributions
    
    # Compute rᵢ = Kᵢ * zᵢ + Bᵢ * z₀ for each scenario
    for i in 1:system.n_scenarios
        device_id = system.device_map[i]
        CUDA.device!(device_id)
        
        # rᵢ = Kᵢ * zᵢ
        mul!(r.z_scenarios[i], system.K_scenarios[i], z.z_scenarios[i])
        
        # rᵢ += Bᵢ * z₀
        CUDA.CUBLAS.gemv!('N', one(T), system.B_coupling[i], z.z0, one(T), r.z_scenarios[i])
    end
    
    return r
end

"""
    matvec(system::ArrowheadSystem{T,INT}, z::ArrowheadVector{T}) -> ArrowheadVector{T}

Non-mutating version of matrix-vector product.

Allocates a new ArrowheadVector for the result.
"""
function matvec(
    system::ArrowheadSystem{T,INT},
    z::ArrowheadVector{T}
) where {T,INT}
    r = ArrowheadVector(system, Val{:zeros})
    matvec!(r, system, z)
    return r
end

