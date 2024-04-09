module Lex_QP_Three_Objectives


using OrderedPreferences

function demo(;verbose = false)

    # Algorithm setting 
    ϵ = 1.0
    κ = 0.1
    max_iterations = 10
    relaxation_mode = :standard
    ρ = 1e-7

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


    # Reconstruct Q,q,c to match the new primal dimension
    original_primal_dim = 3 
    previous_dual_dim0 = 5
    previous_dual_dim1i = 11
    previous_dual_dim1e = 3
    primal_dimension = original_primal_dim + previous_dual_dim0 + previous_dual_dim1i + previous_dual_dim1e

    # if relaxation_mode == :standard
    #     c = zeros(primal_dimension)
    #     Q = zeros(primal_dimension, primal_dimension)
    #     q = zeros(primal_dimension)
    #     P = zeros(primal_dimension, primal_dimension)
    #     p = zeros(primal_dimension)

    # elseif relaxation_mode == :l_infinity
    #     # TODO
    # else
    #     error("Relaxation mode not supported")
    # end

    # c[primal_dimension - previous_dual_dim1e + 1:end] = cₒ
    # Q[1:original_primal_dim, 1:original_primal_dim] = Qₒ
    # q[1:original_primal_dim] = qₒ
    # P[1:original_primal_dim, 1:original_primal_dim] = Pₒ
    # p[1:original_primal_dim] = pₒ


    # Define objective and constraints
    objective(x,θ) = 2*(x[1]^2 + x[2]^2) + p'x[1:3]

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
        c5(x,θ); # x₃
        x[4:8]; # λ¹
        x[9:19]; # λ²
        -x[4]*c1(x,θ) - x[5]*c2(x,θ) - x[6]*c3(x,θ) - x[7]*c4(x,θ) - x[8]*c5(x,θ); # TODO: add ρ
        #ρ - x[4]*c1(x,θ) - x[5]*c2(x,θ) - x[6]*c3(x,θ) - x[7]*c4(x,θ) - x[8]*c5(x,θ); 

    ] 

    complementarity_constraints(x,θ) = [
        -x[9]* x[4]; # λ²ᵢ ⟂ λ¹ᵢ
        -x[10]*x[5];
        -x[11]*x[6];
        -x[12]*x[7];
        -x[13]*x[8]; 
        -x[14]*c1(x,θ); # λ²ᵢ₊₅ ⟂ cᵢ  
        -x[15]*c2(x,θ);
        -x[16]*c3(x,θ);
        -x[17]*c4(x,θ);
        -x[18]*c5(x,θ);
        -x[19]* (- x[4]*c1(x,θ) - x[5]*c2(x,θ) - x[6]*c3(x,θ) - x[7]*c4(x,θ) - x[8]*c5(x,θ)); # TODO: add -x[19]*ρ
        # λ²₁₁ ⟂ (ρ - ...) / -x[19]* (ρ - x[4]*c1(x,θ) - x[5]*c2(x,θ) - x[6]*c3(x,θ) - x[7]*c4(x,θ) - x[8]*c5(x,θ)); 

    ]

    if relaxation_mode == :standard
        equality_constraints(x,θ) = [
            (x[1]*Q[:,1] + x[2]*Q[:,2] + x[3]*Q[:,3] + q - x[14]*[1;-1;-1] - x[15]*[1;1;-1] 
            - x[16]*[-1;1;-1] - x[17]*[-1;-1;-1] - x[18]*[0;0;1] 
            + x[19] *(x[4]*[1;-1;-1] + x[5]*[1;1;-1] + x[6]*[-1;1;-1] + x[7]*[-1;-1;-1] + x[8]*[0;0;1]));

            (-x[20]*[1;1;-1;-1;0] - x[21]*[-1;1;1;-1;0] - x[22]*[-1;-1;-1;-1;1] - [x[9];x[10];x[11];x[12];x[13]] 
            + x[19]*[c1(x,θ); c2(x,θ); c3(x,θ); c4(x,θ); c5(x,θ)]);

            c - x[4]*[1;-1;-1;] - x[5]*[1;1;-1] - x[6]*[-1;1;-1] - x[7]*[-1;-1;-1] - x[8]*[0;0;1];
        ]

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
        relaxation_mode = relaxation_mode,
    )
    elseif relaxation_mode == :l_infinity
        # TODO

        MPCC_prob =  ParametricMPCC(; 
        objective = objective,
        equality_constraints = equality_constraints, # TODO
        inequality_constraints = inequality_constraints,
        complementarity_constraints = complementarity_constraints,
        primal_dimension = primal_dimension,
        parameter_dimension = 1,
        relaxation_parameter = ϵ,
        update_parameter = κ,
        max_iterations = max_iterations,
        relaxation_mode = relaxation_mode,
    )
    else
        error("Relaxation mode not supported")
    end
    
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