# Data structures for arrowhead-structured systems

"""
    ArrowheadSystem{T,INT}

Represents an arrowhead-structured linear system K Δz = r.

The system has the block structure:
```
K = [K₀    B₁ᵀ  B₂ᵀ  ⋯  Bₙᵀ]
    [B₁    K₁   0   ⋯  0  ]
    [B₂    0    K₂  ⋯  0  ]
    [⋮     ⋮    ⋮   ⋱  ⋮  ]
    [Bₙ    0    0   ⋯  Kₙ ]
```

# Fields
- `K0::CuSparseMatrixCSR{T,INT}`: First-stage matrix (n0 × n0)
- `K_scenarios::Vector{CuSparseMatrixCSR{T,INT}}`: Scenario matrices {Kᵢ} (each nᵢ × nᵢ)
- `B_coupling::Vector{CuMatrix{T}}`: Coupling matrices {Bᵢ} (each nᵢ × n0)
- `n0::Int`: Dimension of first-stage variables
- `n_scenarios::Int`: Number of scenarios
- `scenario_dims::Vector{Int}`: Dimensions of each scenario block
- `device_map::Vector{Int}`: GPU device assignment for each scenario (0-based)
- `structure::String`: Matrix structure ("G", "S", "H", "SPD", "HPD")
- `view::Char`: Matrix view ('L', 'U', 'F')
"""
struct ArrowheadSystem{T<:Union{Float32,Float64,ComplexF32,ComplexF64}, INT<:Union{Int32,Int64}, BCT<:CuMatrix{T}}
    K0::CuSparseMatrixCSR{T,INT}
    K_scenarios::Vector{CuSparseMatrixCSR{T,INT}}
    B_coupling::Vector{BCT}
    n0::Int
    n_scenarios::Int
    scenario_dims::Vector{Int}
    device_map::Vector{Int}
    structure::String
    view::Char

    function ArrowheadSystem(
        K0::CuSparseMatrixCSR{T,INT},
        K_scenarios::Vector{CuSparseMatrixCSR{T,INT}},
        B_coupling::Vector{BCT},
        device_map::Vector{Int} = collect(0:length(K_scenarios)-1) .% CUDA.ndevices();
        structure::String = "G",
        view::Char = 'F'
    ) where {T,INT,BCT<:CuMatrix{T}}
        n0 = size(K0, 1)
        n_scenarios = length(K_scenarios)
        scenario_dims = [size(K, 1) for K in K_scenarios]

        # Validate dimensions
        @assert size(K0, 1) == size(K0, 2) "K0 must be square"
        @assert length(K_scenarios) == length(B_coupling) "Must have same number of scenarios and coupling matrices"
        @assert all(size(K, 1) == size(K, 2) for K in K_scenarios) "All scenario matrices must be square"
        @assert all(size(B, 2) == n0 for B in B_coupling) "All coupling matrices must have n0 columns"
        @assert all(size(B_coupling[i], 1) == scenario_dims[i] for i in 1:n_scenarios) "Coupling matrix rows must match scenario dimensions"
        @assert structure in ["G", "S", "H", "SPD", "HPD"] "Invalid structure"
        @assert view in ['L', 'U', 'F'] "Invalid view"

        new{T,INT,BCT}(K0, K_scenarios, B_coupling, n0, n_scenarios, scenario_dims, device_map, structure, view)
    end
end

"""
    ArrowheadVector{T} <: AbstractVector{T}

Represents a distributed vector for the arrowhead system.

# Fields
- `z0::CuVector{T}`: First-stage variables
- `z_scenarios::Vector{CuVector{T}}`: Scenario variables {zᵢ}
- `device_map::Vector{Int}`: GPU device assignment for each scenario component
"""
struct ArrowheadVector{T, VT<:CuVector{T}} <: AbstractVector{T}
    z0::CuVector{T}
    z_scenarios::Vector{VT}
    device_map::Vector{Int}

    function ArrowheadVector(z0::CuVector{T}, z_scenarios::Vector{VT}, device_map::Vector{Int}) where {T, VT<:CuVector{T}}
        @assert length(z_scenarios) == length(device_map) "Number of scenarios must match device map"
        new{T,VT}(z0, z_scenarios, device_map)
    end
end

# Convenience constructor from ArrowheadSystem
function ArrowheadVector(system::ArrowheadSystem{T,INT}, ::Type{Val{:zeros}}) where {T,INT}
    z0 = CUDA.zeros(T, system.n0)
    z_scenarios = [CUDA.zeros(T, dim) for dim in system.scenario_dims]
    ArrowheadVector(z0, z_scenarios, system.device_map)
end

function ArrowheadVector(system::ArrowheadSystem{T,INT}, ::Type{Val{:rand}}) where {T,INT}
    z0 = CUDA.rand(T, system.n0)
    z_scenarios = [CUDA.rand(T, dim) for dim in system.scenario_dims]
    ArrowheadVector(z0, z_scenarios, system.device_map)
end

# Required for Krylov.jl
Base.eltype(::Type{ArrowheadVector{T,VT}}) where {T,VT} = T
Base.eltype(::ArrowheadVector{T}) where T = T

function Base.length(v::ArrowheadVector{T}) where T
    n = length(v.z0)
    for z_scen in v.z_scenarios
        n += length(z_scen)
    end
    return n
end

Base.size(v::ArrowheadVector) = (length(v),)

function Base.similar(v::ArrowheadVector{T}) where T
    z0 = similar(v.z0)
    z_scenarios = [similar(z) for z in v.z_scenarios]
    return ArrowheadVector(z0, z_scenarios, copy(v.device_map))
end

# Required AbstractVector interface
function Base.getindex(v::ArrowheadVector{T}, i::Int) where T
    n0 = length(v.z0)
    if i <= n0
        return v.z0[i]
    end
    idx = i - n0
    for z_scen in v.z_scenarios
        n_scen = length(z_scen)
        if idx <= n_scen
            return z_scen[idx]
        end
        idx -= n_scen
    end
    throw(BoundsError(v, i))
end

function Base.setindex!(v::ArrowheadVector{T}, val, i::Int) where T
    n0 = length(v.z0)
    if i <= n0
        v.z0[i] = val
        return val
    end
    idx = i - n0
    for z_scen in v.z_scenarios
        n_scen = length(z_scen)
        if idx <= n_scen
            z_scen[idx] = val
            return val
        end
        idx -= n_scen
    end
    throw(BoundsError(v, i))
end

"""
    ArrowheadWorkspace{T}

Preallocated workspace arrays for ArrowheadSolver to avoid allocations during solve.

# Fields
- BiCGStab vectors: `x`, `r`, `r_tilde`, `p`, `v`, `s`, `t`, `y`, `z`, `temp`
- Preconditioner workspace: `r_aug_gpus`, `r_tilde_0`
- Krylov.jl workspace: `krylov_workspace` (allocated lazily on first solve)
"""
mutable struct ArrowheadWorkspace{T}
    # Solution and BiCGStab vectors
    x::ArrowheadVector{T}
    r::ArrowheadVector{T}
    r_tilde::ArrowheadVector{T}
    p::ArrowheadVector{T}
    v::ArrowheadVector{T}
    s::ArrowheadVector{T}
    t::ArrowheadVector{T}
    y::ArrowheadVector{T}
    z::ArrowheadVector{T}
    temp::ArrowheadVector{T}

    # Preconditioner workspace (per-GPU augmented vectors)
    r_aug_gpus::Dict{Int,CuVector{T}}
    r_tilde_0::CuVector{T}

    # Krylov.jl workspace (allocated lazily on first solve)
    krylov_workspace::Union{Nothing, Any}

    function ArrowheadWorkspace(system::ArrowheadSystem{T,INT}) where {T,INT}
        # Allocate BiCGStab vectors
        x = ArrowheadVector(system, Val{:zeros})
        r = ArrowheadVector(system, Val{:zeros})
        r_tilde = ArrowheadVector(system, Val{:zeros})
        p = ArrowheadVector(system, Val{:zeros})
        v = ArrowheadVector(system, Val{:zeros})
        s = ArrowheadVector(system, Val{:zeros})
        t = ArrowheadVector(system, Val{:zeros})
        y = ArrowheadVector(system, Val{:zeros})
        z = ArrowheadVector(system, Val{:zeros})
        temp = ArrowheadVector(system, Val{:zeros})

        # Allocate preconditioner workspace
        r_aug_gpus = Dict{Int,CuVector{T}}()
        r_tilde_0 = CUDA.zeros(T, system.n0)

        # Krylov workspace allocated lazily on first solve
        krylov_workspace = nothing

        new{T}(x, r, r_tilde, p, v, s, t, y, z, temp, r_aug_gpus, r_tilde_0, krylov_workspace)
    end
end

"""
    ArrowheadSolver{T,INT} <: Factorization{T}

Solver for arrowhead-structured linear systems following LinearAlgebra API.

Similar to `lu` or `cholesky`, this represents a factorized arrowhead system
that can be used to efficiently solve multiple right-hand sides.

# Fields
- `system::ArrowheadSystem{T,INT}`: The arrowhead system structure
- Factorization data: `gpu_solvers`, `schur_complements`, `C_aggregated`, `C_factors`
- `workspace::ArrowheadWorkspace{T}`: Preallocated arrays for efficient solving
- BiCGStab parameters: `tol`, `maxiter`, `verbose`
- `factorized::Bool`: Whether the system has been factorized

# Usage
```julia
# Create and factorize (like lu(A))
system = ArrowheadSystem(K0, K_scenarios, B_coupling, device_map)
solver = factorize(system)

# Solve (like F \\ b)
x = solver \\ b              # allocating
ldiv!(x, solver, b)         # in-place: x = solver \\ b
ldiv!(solver, b)            # in-place: b = solver \\ b (overwrites b)

# Update and refactorize (like lu!(F, A))
factorize(solver, system_new)
```
"""
mutable struct ArrowheadSolver{T,INT,BCT} <: Factorization{T}
    # System data
    system::ArrowheadSystem{T,INT,BCT}

    # Factorization data
    gpu_solvers::Dict{Int,Any}  # GPU device_id => augmented CUDSS solver
    scenario_to_gpu::Dict{Int,Int}  # scenario index => GPU device_id
    gpu_to_scenarios::Dict{Int,Vector{Int}}  # GPU device_id => scenario indices
    schur_complements::Vector{CuMatrix{T}}  # One per GPU
    C_aggregated::Union{Nothing,CuMatrix{T}}
    C_factors::Union{Nothing,Any}  # DenseLUFactors{T}

    # Workspace arrays (preallocated)
    workspace::ArrowheadWorkspace{T}

    # BiCGStab parameters
    tol::Float64
    maxiter::Int
    verbose::Bool

    # Factorization state
    factorized::Bool

    function ArrowheadSolver(
        system::ArrowheadSystem{T,INT,BCT};
        tol::Float64 = 1e-6,
        maxiter::Int = 100,
        verbose::Bool = false
    ) where {T,INT,BCT}
        # Build GPU mappings
        scenario_to_gpu = Dict{Int,Int}()
        gpu_to_scenarios = Dict{Int,Vector{Int}}()

        for i in 1:system.n_scenarios
            gpu_id = system.device_map[i]
            scenario_to_gpu[i] = gpu_id

            if !haskey(gpu_to_scenarios, gpu_id)
                gpu_to_scenarios[gpu_id] = Int[]
            end
            push!(gpu_to_scenarios[gpu_id], i)
        end

        n_gpus = length(gpu_to_scenarios)
        gpu_solvers = Dict{Int,Any}()
        schur_complements = Vector{CuMatrix{T}}(undef, n_gpus)

        # Allocate workspace
        workspace = ArrowheadWorkspace(system)

        new{T,INT,BCT}(
            system,
            gpu_solvers, scenario_to_gpu, gpu_to_scenarios,
            schur_complements, nothing, nothing,
            workspace,
            tol, maxiter, verbose,
            false  # not yet factorized
        )
    end
end

# Type aliases for backward compatibility
const ArrowheadFactors{T,INT} = ArrowheadSolver{T,INT}

# Vector operations
function LinearAlgebra.dot(x::ArrowheadVector{T}, y::ArrowheadVector{T}) where T
    result = dot(x.z0, y.z0)
    for i in 1:length(x.z_scenarios)
        result += dot(x.z_scenarios[i], y.z_scenarios[i])
    end
    return result
end

function LinearAlgebra.norm(x::ArrowheadVector{T}) where T
    return sqrt(dot(x, x))
end

function LinearAlgebra.axpy!(α::Number, x::ArrowheadVector{T}, y::ArrowheadVector{T}) where T
    CUDA.axpy!(α, x.z0, y.z0)
    for i in 1:length(x.z_scenarios)
        CUDA.axpy!(α, x.z_scenarios[i], y.z_scenarios[i])
    end
    return y
end

function LinearAlgebra.axpby!(α::Number, x::ArrowheadVector{T}, β::Number, y::ArrowheadVector{T}) where T
    # y = α*x + β*y
    CUDA.axpby!(α, x.z0, β, y.z0)
    for i in 1:length(x.z_scenarios)
        CUDA.axpby!(α, x.z_scenarios[i], β, y.z_scenarios[i])
    end
    return y
end

function Base.copy!(dst::ArrowheadVector{T}, src::ArrowheadVector{T}) where T
    copyto!(dst.z0, src.z0)
    for i in 1:length(src.z_scenarios)
        copyto!(dst.z_scenarios[i], src.z_scenarios[i])
    end
    return dst
end

# Override copyto! to avoid scalar indexing
function Base.copyto!(dst::ArrowheadVector{T}, src::ArrowheadVector{T}) where T
    copy!(dst, src)
    return dst
end

# Override fill! to avoid scalar indexing
function Base.fill!(x::ArrowheadVector{T}, val) where T
    fill!(x.z0, val)
    for z_scen in x.z_scenarios
        fill!(z_scen, val)
    end
    return x
end

# Override ldiv! for identity matrix to avoid scalar indexing (needed for Krylov.jl)
function LinearAlgebra.ldiv!(y::ArrowheadVector{T}, ::UniformScaling, x::ArrowheadVector{T}) where T
    copy!(y, x)  # Identity operation: y = I⁻¹x = x
    return y
end

function Base.copy(src::ArrowheadVector{T,VT}) where {T,VT}
    z0_copy = copy(src.z0)
    z_scenarios_copy = [copy(z) for z in src.z_scenarios]
    ArrowheadVector(z0_copy, z_scenarios_copy, copy(src.device_map))
end

# Zero out a vector
function zero!(x::ArrowheadVector{T,VT}) where {T,VT}
    fill!(x.z0, zero(T))
    for i in 1:length(x.z_scenarios)
        fill!(x.z_scenarios[i], zero(T))
    end
    return x
end

# Subtraction
function Base.:-(x::ArrowheadVector{T,VT}, y::ArrowheadVector{T,VT}) where {T,VT}
    z0 = x.z0 - y.z0
    z_scenarios = [x.z_scenarios[i] - y.z_scenarios[i] for i in 1:length(x.z_scenarios)]
    ArrowheadVector(z0, z_scenarios, copy(x.device_map))
end

# Scalar multiplication
function Base.:*(α::Number, x::ArrowheadVector{T,VT}) where {T,VT}
    z0 = α * x.z0
    z_scenarios = [α * z for z in x.z_scenarios]
    ArrowheadVector(z0, z_scenarios, copy(x.device_map))
end


