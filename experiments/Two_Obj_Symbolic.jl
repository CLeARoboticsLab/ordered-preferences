module Two_Obj_Symbolic


using OrderedPreferences
using Makie, CairoMakie

function demo(;verbose = false)
    # Lex Min QP Fiaschi (2021)
    # Two objectives:
    Q = [10 -2 4; 
        -2 10 4; 
        4 4 4]  
    q = [-16; -16; -16]
    c = [-1; -1; 0]

    # Define objective and constraints
    objective(x,θ) = c'x[1:3] 

    c1(x,θ) =  x[1] - x[2] - x[3] + 1
    c2(x,θ) =  x[1] + x[2] - x[3] + 1
    c3(x,θ) = -x[1] + x[2] - x[3] + 1
    c4(x,θ) = -x[1] - x[2] - x[3] + 3
    c5(x,θ) =  x[3] 
    # c6(x,θ) = -x[1] - x[2] - x[3] + 1 #0422 test: c'x + 1 ≥ 0
    inequality_constraints(x,θ) = [
        c1(x,θ);
        c2(x,θ);
        c3(x,θ);
        c4(x,θ);
        c5(x,θ);
        # c6(x,θ);
    ] 

    equality_constraints(x,θ) = []

    prioritized_preferences = [  
        function (x, θ)
            0.5*x[1:3]'Q*x[1:3] + q'x[1:3]
        end,
    ]
    is_prioritized_constraint = [false]

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

end


end # module end