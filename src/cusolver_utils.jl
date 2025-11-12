# cuSOLVER utilities for dense matrix operations

using CUDA
using CUDA.CUSOLVER
using LinearAlgebra

"""
    DenseLUFactors{T}

Holds the LU factorization of a dense matrix.

# Fields
- `lu_fact`: LinearAlgebra.LU factorization object from CUDA
- `n::Int`: Matrix dimension
"""
struct DenseLUFactors{T}
    lu_fact::LinearAlgebra.LU{T, <:CuMatrix{T}, <:CuVector}
    n::Int
    
    function DenseLUFactors{T}(lu_fact::LinearAlgebra.LU{T}, n::Int) where T
        new{T}(lu_fact, n)
    end
    
    function DenseLUFactors(lu_fact::LinearAlgebra.LU{T}, n::Int) where T
        new{T}(lu_fact, n)
    end
end

"""
    dense_lu!(A::CuMatrix{T}) -> DenseLUFactors{T}

Compute LU factorization of a dense matrix using LinearAlgebra.lu.

The factorization is of the form P*A = L*U, where P is a permutation matrix,
L is lower triangular with unit diagonal, and U is upper triangular.

# Arguments
- `A::CuMatrix{T}`: Dense matrix to factorize

# Returns
- `DenseLUFactors{T}`: Structure containing the factors and pivot information
"""
function dense_lu!(A::CuMatrix{T}) where T<:Union{Float32,Float64,ComplexF32,ComplexF64}
    n, m = size(A)
    @assert n == m "Matrix must be square for LU factorization"
    
    # Use LinearAlgebra's lu function which works with CuArrays
    lu_fact = LinearAlgebra.lu(A)
    
    return DenseLUFactors(lu_fact, n)
end

"""
    dense_lu_solve!(factors::DenseLUFactors{T}, B::CuMatrix{T})

Solve the system A*X = B using pre-computed LU factors.

# Arguments
- `factors::DenseLUFactors{T}`: Pre-computed LU factorization
- `B::CuMatrix{T}`: Right-hand side matrix (will be overwritten with solution)

# Returns
- `X::CuMatrix{T}`: Solution matrix
"""
function dense_lu_solve!(factors::DenseLUFactors{T}, B::CuMatrix{T}) where T
    n, nrhs = size(B)
    @assert n == factors.n "Dimension mismatch between factors and RHS"
    
    # Solve using the LU factorization (ldiv! operates in-place)
    X = factors.lu_fact \ B
    
    return X
end

"""
    dense_lu_solve!(factors::DenseLUFactors{T}, b::CuVector{T})

Solve the system A*x = b using pre-computed LU factors (vector version).

# Arguments
- `factors::DenseLUFactors{T}`: Pre-computed LU factorization
- `b::CuVector{T}`: Right-hand side vector (will be overwritten with solution)

# Returns
- `b::CuVector{T}`: Solution vector (same as input, modified in-place)
"""
function dense_lu_solve!(factors::DenseLUFactors{T}, b::CuVector{T}) where T
    # Convert vector to matrix for cuSOLVER call
    B = reshape(b, (length(b), 1))
    dense_lu_solve!(factors, B)
    return b
end

