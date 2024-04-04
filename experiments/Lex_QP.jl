module Lex_QP


using OrderedPreferences

function demo(;verbose = false)

    # Lex Min QP Fiaschi (2021)
    Qₒ = [10 -2 4; 
         -2 10 4; 
          4 4 4]  
    qₒ = [-16; -16; -16]
    cₒ = [-1; -1; 0]

    # Reconstruct Q,q,c to match the new primal dimension 
    Q = zeros(8,8)
    Q[1:3, 1:3] = Qₒ

    q = zeros(8)
    q[1:3] = qₒ

    c = zeros(8)
    c[1:3] = cₒ

    # Define objective and constraints
    objective(x,θ) = c'x

    equality_constraints(x,θ) = [
        Q*x + q - x[4]*[1;-1;-1;zeros(5)] - x[5]*[1;1;-1;zeros(5)] - x[6]*[-1;1;-1;zeros(5)] - x[7]*[-1;-1;-1;zeros(5)] - x[8]*[0;0;1;zeros(5)];
    ]
    
    c1(x,θ) =  x[1] - x[2] - x[3] + 1
    c2(x,θ) =  x[1] + x[2] - x[3] + 1
    c3(x,θ) = -x[1] + x[2] - x[3] + 1
    c4(x,θ) = -x[1] - x[2] - x[3] + 3
    inequality_constraints(x,θ) = [
        c1(x,θ);
        c2(x,θ);
        c3(x,θ);
        c4(x,θ);
        x[3];
    ] 

    complementarity_constraints(x,θ) = [
        -x[4]*c1(x,θ);
        -x[5]*c2(x,θ);
        -x[6]*c3(x,θ);
        -x[7]*c4(x,θ);
        -x[8]*x[3];
    ]
    primal_dimension = 3 + 5





    ϵ = 1.0
    κ = 0.1
    max_iterations = 10
    
    MPCC_prob =  ParametricMPCC(; 
        objective = objective,
        equality_constraints = equality_constraints,
        inequality_constraints = inequality_constraints,
        complementarity_constraints = complementarity_constraints,
        primal_dimension = primal_dimension,
        parameter_dimension = 1,
        relaxation_parameter = ϵ,
        update_parameter = κ,
        max_iterations = max_iterations,
        relaxation_mode = :l_infinity,
    )

    # Solve MPCC
    tolerance = 1e-7
    parameters = zeros(MPCC_prob.subproblems[1].parameter_dimension)
    (; relaxation, solution, residual) = 
        solve_relaxed_mpcc(MPCC_prob, nothing, parameters; ϵ, κ, max_iterations, tolerance, verbose)
    println("relaxation: ", relaxation)
    println("solution: ", solution)
    println("residual: ", residual)
end


end # module end