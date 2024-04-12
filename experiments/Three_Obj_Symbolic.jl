module Three_Obj_Symbolic


using OrderedPreferences

function demo(;verbose = false)

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
            c'x[1:3]
        end,
        
        function (x, θ)
            0.5*x[1:3]'Q*x[1:3] + q'x[1:3]
        end,
    ]

    # Problem setting 
    primal_dimension = 3

    preferences_dimension = length(prioritized_preferences)
    parameters = zeros(preferences_dimension - 1) 
    parameter_dimension = length(parameters) #same as no. of intermediate inner levels 
    equality_dimension = length(equality_constraints(zeros(primal_dimension), zeros(parameter_dimension)))
    inequality_dimension = length(inequality_constraints(zeros(primal_dimension), zeros(parameter_dimension)))
    
    # Algorithm setting 
    ϵ = 1.0
    κ = 0.1
    max_iterations = 10
    tolerance = 1e-7
    relaxation_mode = :standard
    println("relaxation_mode: ", relaxation_mode)

    POP_prob = ParametricOrderedPreferencesMPCC(;
        objective,
        equality_constraints,
        inequality_constraints,
        prioritized_preferences,
        primal_dimension,
        parameter_dimension,
        equality_dimension,
        inequality_dimension,
        relaxation_parameter = ϵ,
        update_parameter = κ,
        max_iterations,
        relaxation_mode,
    )

    # Solve POP
    (; relaxation, solution, residual) = 
        solve_relaxed_mpcc(POP_prob, nothing, parameters; ϵ, κ, max_iterations, tolerance, verbose)
    println("relaxation: ", relaxation)
    println("solution: ", solution)
    println("residual: ", residual)
end


end # module end