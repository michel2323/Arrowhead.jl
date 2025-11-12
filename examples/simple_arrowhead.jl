"""
Example: Solving an arrowhead-structured linear system using CUDSS

This example demonstrates:
1. Creating an arrowhead system with multiple scenarios
2. Factorizing the system using Schur complement method
3. Solving using BiCGStab with preconditioning
4. Using multiple GPUs for scenario distribution
"""

using Arrowhead
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR
using LinearAlgebra
using SparseArrays

println("="^70)
println("Arrowhead Solver Example")
println("="^70)
println()

# Check CUDA availability
if !CUDA.functional()
    error("CUDA is not functional. This example requires a GPU.")
end

println("CUDA Information:")
println("  Number of GPUs available: $(CUDA.ndevices())")
for i in 1:CUDA.ndevices()
    CUDA.device!(i-1)
    println("  GPU $(i-1): $(CUDA.name(CUDA.device()))")
end
println()

# Problem parameters
n0 = 5               # First-stage dimension
n_scenarios = 4      # Number of scenarios
n_scenario = 20      # Dimension of each scenario
coupling_strength = 0.05  # Strength of coupling between stages (small for stability)

println("Problem Setup:")
println("  First-stage dimension: $n0")
println("  Number of scenarios: $n_scenarios")
println("  Scenario dimension: $n_scenario")
println("  Total system size: $(n0 + n_scenarios * n_scenario)")
println()

# Create first-stage matrix (symmetric positive definite)
println("Creating first-stage matrix K₀...")
A0 = sprand(Float64, n0, n0, 0.8)
A0 = A0 + A0' + 5.0 * I  # Make SPD
K0 = CuSparseMatrixCSR(A0)

# Create scenario matrices and coupling matrices
println("Creating scenario matrices and coupling...")
K_scenarios = CuSparseMatrixCSR{Float64,Int32}[]
B_coupling = CuMatrix{Float64}[]

for i in 1:n_scenarios
    # Create SPD scenario matrix
    A_scen = sprand(Float64, n_scenario, n_scenario, 0.5)
    A_scen = A_scen + A_scen' + 3.0 * I
    push!(K_scenarios, CuSparseMatrixCSR(A_scen))

    # Create coupling matrix
    B = CUDA.randn(Float64, n_scenario, n0) * coupling_strength
    push!(B_coupling, B)
end

# Create device mapping for multi-GPU (if available)
if CUDA.ndevices() > 1
    device_map = [i % CUDA.ndevices() for i in 0:n_scenarios-1]
    println("Multi-GPU mode: distributing scenarios across $(CUDA.ndevices()) GPUs")
    println("Device mapping: $device_map")
else
    device_map = zeros(Int, n_scenarios)
    println("Single-GPU mode")
end
println()

# Create arrowhead system
println("Creating arrowhead system...")
system = ArrowheadSystem(
    K0, K_scenarios, B_coupling, device_map;
    structure="SPD",  # Symmetric positive definite
    view='U'          # Upper triangular view
)

# Create right-hand side
println("Creating right-hand side vector...")
b = ArrowheadVector(system, Val{:rand})
b_norm = norm(b)
println("  ||b|| = $b_norm")
println()

# Factorize and solve the system using new API
println("="^70)
println("Factorization Phase")
println("="^70)
# Set default tolerance and maxiter for the solver
# Note: For Krylov.jl, atol and rtol are set to the same value
solver = factorize(system; tol=1e-8, maxiter=100, verbose=true)
println()

# Solve the system using solver API
println("="^70)
println("Solution Phase")
println("="^70)
println("Using default solver tolerances (tol=1e-8, maxiter=100):")
x, stats = solver \ b

# Can also use solve! to override tolerances for specific solves:
# x, stats = solve!(solver, b; tol=1e-10, maxiter=200)

# Verify the solution
println("\n" * "="^70)
println("Verification")
println("="^70)

# Compute residual: r = b - K*x
r = copy(b)
Ax = matvec(system, x)
axpy!(-1.0, Ax, r)

residual_norm = norm(r)
relative_residual = residual_norm / b_norm

println("Solution verification:")
println("  ||b|| = $b_norm")
println("  ||x|| = $(norm(x))")
println("  ||r|| = $residual_norm")
println("  ||r|| / ||b|| = $relative_residual")
println("  Bicgstab iterations = $(stats.niter)")
println()

if relative_residual < 1e-5
    println("✓ Solution PASSED: residual is within tolerance (< 1e-5)")
else
    println("✗ Solution FAILED: residual is too large (>= 1e-5)")
end

println()
println("="^70)
println("Example Complete")
println("="^70)

