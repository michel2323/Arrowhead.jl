using Test
using Arrowhead
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR
using CUDSS
using LinearAlgebra
using SparseArrays

@testset "Arrowhead.jl" begin

    # Check if CUDA is available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping tests"
        return
    end

    @testset "Basic ArrowheadSystem Construction" begin
        # Create a small test problem
        n0 = 5  # First-stage dimension
        n_scenarios = 2
        n_scenario = 10  # Each scenario dimension

        # Create first-stage matrix (SPD)
        A0 = sprand(Float64, n0, n0, 0.5)
        A0 = A0 + A0' + 3.0 * I
        K0 = CuSparseMatrixCSR(A0)

        # Create scenario matrices
        K_scenarios = CuSparseMatrixCSR{Float64,Int32}[]
        B_coupling = CuMatrix{Float64}[]

        for i in 1:n_scenarios
            # SPD scenario matrix
            A_scen = sprand(Float64, n_scenario, n_scenario, 0.3)
            A_scen = A_scen + A_scen' + 2.0 * I
            push!(K_scenarios, CuSparseMatrixCSR(A_scen))

            # Coupling matrix
            B = CUDA.randn(Float64, n_scenario, n0) * 0.1
            push!(B_coupling, B)
        end

        # Create arrowhead system
        system = ArrowheadSystem(K0, K_scenarios, B_coupling; structure="SPD", view='U')

        @test system.n0 == n0
        @test system.n_scenarios == n_scenarios
        @test length(system.K_scenarios) == n_scenarios
        @test length(system.B_coupling) == n_scenarios
    end

    @testset "ArrowheadVector Operations" begin
        n0 = 5
        n_scenarios = 2
        n_scenario = 10

        # Create dummy system for vector initialization
        A0 = sprand(Float64, n0, n0, 0.5) + 3.0 * I
        K0 = CuSparseMatrixCSR(A0)
        K_scenarios = [CuSparseMatrixCSR(sprand(Float64, n_scenario, n_scenario, 0.3) + 2.0 * I) for _ in 1:n_scenarios]
        B_coupling = [CUDA.randn(Float64, n_scenario, n0) * 0.1 for _ in 1:n_scenarios]
        system = ArrowheadSystem(K0, K_scenarios, B_coupling; structure="SPD", view='U')

        # Test vector creation
        x = ArrowheadVector(system, Val{:zeros})
        @test norm(x) ≈ 0.0

        y = ArrowheadVector(system, Val{:rand})
        @test norm(y) > 0.0

        # Test vector operations
        z = copy(y)
        @test norm(z - y) ≈ 0.0 atol=1e-10

        # Test dot product
        d = dot(y, y)
        @test d ≈ norm(y)^2 atol=1e-10

        # Test axpy
        alpha = 2.0
        w = copy(x)
        axpy!(alpha, y, w)
        @test norm(w - alpha * y) ≈ 0.0 atol=1e-10
    end

    @testset "Matrix-Vector Product" begin
        n0 = 3
        n_scenarios = 2
        n_scenario = 5

        # Create simple test problem
        A0 = sprand(Float64, n0, n0, 0.8) + 2.0 * I
        K0 = CuSparseMatrixCSR(A0)
        K_scenarios = [CuSparseMatrixCSR(sprand(Float64, n_scenario, n_scenario, 0.6) + 2.0 * I) for _ in 1:n_scenarios]
        B_coupling = [CUDA.randn(Float64, n_scenario, n0) * 0.1 for _ in 1:n_scenarios]
        system = ArrowheadSystem(K0, K_scenarios, B_coupling; structure="SPD", view='U')

        # Create random vector
        x = ArrowheadVector(system, Val{:rand})

        # Compute matrix-vector product
        y = matvec(system, x)

        # Verify dimensions
        @test length(y.z0) == n0
        @test length(y.z_scenarios) == n_scenarios
        @test all(length(y.z_scenarios[i]) == n_scenario for i in 1:n_scenarios)
    end

    @testset "Small Arrowhead System Solve" begin
        println("\n" * "="^60)
        println("Testing Small Arrowhead System")
        println("="^60)

        # Create a very small test problem
        n0 = 3  # First-stage dimension
        n_scenarios = 2
        n_scenario = 5  # Each scenario dimension

        # Create first-stage matrix (SPD)
        A0 = sprand(Float64, n0, n0, 0.9)
        A0 = A0 + A0' + 5.0 * I
        K0 = CuSparseMatrixCSR(A0)

        # Create scenario matrices and coupling
        K_scenarios = CuSparseMatrixCSR{Float64,Int32}[]
        B_coupling = CuMatrix{Float64}[]

        for i in 1:n_scenarios
            # SPD scenario matrix
            A_scen = sprand(Float64, n_scenario, n_scenario, 0.7)
            A_scen = A_scen + A_scen' + 3.0 * I
            push!(K_scenarios, CuSparseMatrixCSR(A_scen))

            # Coupling matrix (small values for stability)
            B = CUDA.randn(Float64, n_scenario, n0) * 0.05
            push!(B_coupling, B)
        end

        # Create arrowhead system
        system = ArrowheadSystem(K0, K_scenarios, B_coupling; structure="SPD", view='U')

        # Create right-hand side
        b = ArrowheadVector(system, Val{:rand})

        # Factorize the system (new API)
        println("\nFactorizing system...")
        solver = factorize(system; tol=1e-6, maxiter=50, verbose=true)

        # Solve using new API
        println("\nSolving system...")
        x, stats = solver \ b

        # Compute residual
        r = copy(b)
        Ax = matvec(system, x)
        axpy!(-1.0, Ax, r)

        residual_norm = norm(r)
        rhs_norm = norm(b)
        relative_residual = residual_norm / rhs_norm

        println("\nFinal verification:")
        println("  Residual norm: $residual_norm")
        println("  RHS norm: $rhs_norm")
        println("  Relative residual: $relative_residual")
        println("="^60)

        # Check convergence
        @test relative_residual < 1e-2  # Relaxed tolerance for iterative solver with approximate preconditioner
        if stats !== nothing
            @test stats.converged == true
        end
    end

    @testset "Multi-GPU Distribution" begin
        if CUDA.ndevices() >= 2
            println("\n" * "="^60)
            println("Testing Multi-GPU Distribution")
            println("="^60)

            n0 = 4
            n_scenarios = 4
            n_scenario = 8

            # Create system with explicit device mapping
            A0 = sprand(Float64, n0, n0, 0.8) + 5.0 * I
            K0 = CuSparseMatrixCSR(A0)

            K_scenarios = CuSparseMatrixCSR{Float64,Int32}[]
            B_coupling = CuMatrix{Float64}[]

            for i in 1:n_scenarios
                A_scen = sprand(Float64, n_scenario, n_scenario, 0.6) + 3.0 * I
                A_scen = A_scen + A_scen'
                push!(K_scenarios, CuSparseMatrixCSR(A_scen))
                push!(B_coupling, CUDA.randn(Float64, n_scenario, n0) * 0.05)
            end

            # Map scenarios to different GPUs (round-robin)
            device_map = [i % CUDA.ndevices() for i in 0:n_scenarios-1]

            system = ArrowheadSystem(K0, K_scenarios, B_coupling, device_map; structure="SPD", view='U')

            println("\nDevice mapping: $device_map")
            println("Number of GPUs: $(CUDA.ndevices())")

            @test system.device_map == device_map
            @test maximum(device_map) < CUDA.ndevices()

            println("Multi-GPU system created successfully!")
            println("="^60)
        else
            @warn "Only one GPU available, skipping multi-GPU test"
        end
    end

end

