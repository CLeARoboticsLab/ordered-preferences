using OrderedPreferences
using Test: @testset, @test

include("../experiments/SimpleLinearExample.jl")

@testset "All tests" begin
    # @testset "OptimizationTests" begin
    #     f(x, θ) = sum(x)
    #     g(x, θ) = [sum(x .^ 2) - 1]
    #     h(x, θ) = -x

    #     problem = ParametricOptimizationProblem(;
    #         objective = f,
    #         equality_constraint = g,
    #         inequality_constraint = h,
    #         parameter_dimension = 1,
    #         primal_dimension = 2,
    #         equality_dimension = 1,
    #         inequality_dimension = 2,
    #     )

    #     (; primals, variables, status, info) = solve(problem)
    #     println("primals: ", primals)
    #     println("variables: ", variables)
    #     println("status: ", status)
    #     println("info: ", info)
    #     @test all(isapprox.(primals, -0.5sqrt(2), atol = 1e-6))
    # end
    #
    # @testset "Simple linear ordered preferences problem" begin
    #     ordered_preferences_problem = SimpleLinearExample.get_problem()
    #     solution = solve(ordered_preferences_problem)
    #     @test isapprox(solution.primals, [6, 5])
    # end

    # @testset "Two Objective Lex QP Symbolic" begin
    #     # Modified two-obj Lex QP (one priority level)
    #     c = [-1; -1; -1]
    #     Q = [2 2 0; 
    #         2 2 0; 
    #         0 0 4]  
    #     q = [-5; -5; 0]
    #     P = [4 0 0; 
    #         0 4 0; 
    #         0 0 0]  
    #     p = [-5; -3; 2]

    #     # Define objective and constraints
    #     objective(x,θ) = 0.5*x[1:3]'P*x[1:3] + p'x[1:3]


    #     c1(x,θ) =  x[1] - x[2] - x[3] + 1
    #     c2(x,θ) =  x[1] + x[2] - x[3] + 1
    #     c3(x,θ) = -x[1] + x[2] - x[3] + 1
    #     c4(x,θ) = -x[1] - x[2] - x[3] + 3
    #     c5(x,θ) =  x[3] 
    #     c6(x,θ) = -x[1] - x[2] - x[3] + 1 #0422 test: c'x + 1 ≥ 0
    #     inequality_constraints(x,θ) = [
    #         c1(x,θ);
    #         c2(x,θ);
    #         c3(x,θ);
    #         c4(x,θ);
    #         c5(x,θ);
    #         c6(x,θ);
    #     ] 

    #     equality_constraints(x,θ) = []

    #     prioritized_preferences = [  
    #         function (x, θ)
    #             0.5*x[1:3]'Q*x[1:3] + q'x[1:3]
    #         end,
    #     ]
    #     is_prioritized_constraint = [false]

    #     # Problem setting 
    #     primal_dimension = 3
    #     parameter_dimension = 2 # stay user-defined
    #     parameters = ones(parameter_dimension) # dummy 

    #     equality_dimension = length(equality_constraints(zeros(primal_dimension), zeros(parameter_dimension)))
    #     inequality_dimension = length(inequality_constraints(zeros(primal_dimension), zeros(parameter_dimension)))

    #     # Algorithm setting
    #     verbose = true 
    #     ϵ = 1.0
    #     κ = 0.1
    #     max_iterations = 10
    #     tolerance = 1e-6
    #     relaxation_mode = :standard
    #     println("relaxation_mode: ", relaxation_mode)

    #     POP_prob = ParametricOrderedPreferencesMPCC(; # Stay parametrized by θ where θ(end) is the relaxation parameter
    #         objective,
    #         equality_constraints,
    #         inequality_constraints,
    #         prioritized_preferences,
    #         is_prioritized_constraint,
    #         primal_dimension,
    #         parameter_dimension,
    #         equality_dimension,
    #         inequality_dimension,
    #         relaxation_mode,
    #     )

    #     # Solve POP
    #     (; relaxation, solution, residual) = 
    #         solve_relaxed_pop(POP_prob, nothing, parameters; ϵ, κ, max_iterations, tolerance, verbose)
    #     println("relaxation: ", relaxation)
    #     println("solution: ", solution[end]) #TODO: solution
    #     println("residual: ", residual)
    # end

    @testset "Three Objective Lex QP Symbolic" begin 
        # Lex Min QP Fiaschi (2021)
        # Three objectives:
        c = [-1; -1; -1]
        Q = [2 2 0; 
            2 2 0; 
            0 0 4]  
        q = [-5; -5; 0]
        P = [4 0 0; 
            0 4 0; 
            0 0 0]  
        p = [-5; -3; 2]

        # Define objective and constraints
        objective(x,θ) = 0.5*x[1:3]'P*x[1:3] + p'x[1:3] #2*(x[1]^2 + x[2]^2)

        c1(x,θ) =  x[1] - x[2] - x[3] + 1
        c2(x,θ) =  x[1] + x[2] - x[3] + 1
        c3(x,θ) = -x[1] + x[2] - x[3] + 1
        c4(x,θ) = -x[1] - x[2] - x[3] + 3
        c5(x,θ) =  x[3] 
        inequality_constraints(x,θ) = [
            c1(x,θ);
            c2(x,θ);
            c3(x,θ);
            c4(x,θ);
            c5(x,θ);
        ] 

        equality_constraints(x,θ) = []

        prioritized_preferences = [
            function (x, θ)
                c'x[1:3] + 1
            end,
            
            function (x, θ)
                0.5*x[1:3]'Q*x[1:3] + q'x[1:3]
            end,
        ]
        is_prioritized_constraint = [true, false]

        # Problem setting 
        primal_dimension = 3
        parameter_dimension = 2 # stay user-defined
        parameters = ones(parameter_dimension) # dummy 

        equality_dimension = length(equality_constraints(zeros(primal_dimension), zeros(parameter_dimension)))
        inequality_dimension = length(inequality_constraints(zeros(primal_dimension), zeros(parameter_dimension)))
        
        # Algorithm setting
        verbose = true 
        ϵ = 1.0
        κ = 0.1
        max_iterations = 10
        tolerance = 1e-6
        relaxation_mode = :standard
        println("relaxation_mode: ", relaxation_mode)

        POP_prob = ParametricOrderedPreferencesMPCC(; # Stay parametrized by θ where θ(end) is the relaxation parameter
            objective,
            equality_constraints,
            inequality_constraints,
            prioritized_preferences,
            is_prioritized_constraint,
            primal_dimension,
            parameter_dimension,
            equality_dimension,
            inequality_dimension,
            relaxation_mode,
        )

        # Solve POP
        (; relaxation, solution, residual) = 
            solve_relaxed_pop(POP_prob, nothing, parameters; ϵ, κ, max_iterations, tolerance, verbose)
        println("relaxation: ", relaxation)
        println("solution: ", solution[end]) #TODO: solution
        println("residual: ", residual)
        @test all(isapprox.(solution[end].primals[1:primal_dimension], [0.75, 0.25, 0.0], atol = 1e-6))
    
    end


end
