module QP_MPCC

using LinearAlgebra: I

using OrderedPreferences

function demo(;verbose = false)

"""
    # Define QP_MPCC (Reformulation 1: no conversion of inequality constraints to equality constraints)
    objective(x,θ) = (x[1]-5)^2 + (2*x[2]+1)^2

    equality_constraints(x,θ) = [
        2*(x[2]-1) - 1.5*x[1] + x[3] - 0.5*x[4] + x[5];
    ]
    
    c1(x,θ) = 3*x[1] - x[2] - 3
    c2(x,θ) = -x[1] + 0.5*x[2] + 4
    c3(x,θ) = -x[1] - x[2] + 7
    inequality_constraints(x,θ) = [
        x;
        c1(x,θ);
        c2(x,θ);
        c3(x,θ);
    ] 

    complementarity_constraints(x,θ) = [
        -x[3]*c1(x,θ);
        -x[4]*c2(x,θ);
        -x[5]*c3(x,θ);
    ]
    primal_dimension = 5
"""
"""
    # Define QP_MPCC (Reformulation 2: using slacks to conert inequality constraints to equality constraints)
    objective(x,θ) = (x[1]-5)^2 + (2*x[2]+1)^2

    equality_constraints(x,θ) = [
        2*(x[2]-1) - 1.5*x[1] + x[3] - 0.5*x[4] + x[5];
        3*x[1] - x[2] - 3 - x[6];
        -x[1] + 0.5*x[2] + 4 - x[7];
        -x[1] - x[2] + 7 - x[8];
    ]

    inequality_constraints(x,θ) = [
        x;
    ] 

    complementarity_constraints(x,θ) = [
        -x[3]*x[6];
        -x[4]*x[7];
        -x[5]*x[8];
    ]
    primal_dimension = 8
"""
"""
    # Define QP_MPCC (Reformulation 3-1: replace complementarity constraints with a single inequality G(z)ᵀH(z) ≤ 0)
    objective(x,θ) = (x[1]-5)^2 + (2*x[2]+1)^2

    equality_constraints(x,θ) = [
        2*(x[2]-1) - 1.5*x[1] + x[3] - 0.5*x[4] + x[5];
    ]
    
    c1(x,θ) = 3*x[1] - x[2] - 3
    c2(x,θ) = -x[1] + 0.5*x[2] + 4
    c3(x,θ) = -x[1] - x[2] + 7
    inequality_constraints(x,θ) = [
        x;
        c1(x,θ);
        c2(x,θ);
        c3(x,θ);
    ] 

    complementarity_constraints(x,θ) = [-(x[3]*c1(x,θ) + x[4]*c2(x,θ) + x[5]*c3(x,θ))]
    primal_dimension = 5
"""

    # Define QP_MPCC (Reformulation 3-2: replace complementarity constraints with a single inequality G(z)ᵀH(z) ≤ 0)
    objective(x,θ) = (x[1]-5)^2 + (2*x[2]+1)^2

    equality_constraints(x,θ) = [
        2*(x[2]-1) - 1.5*x[1] + x[3] - 0.5*x[4] + x[5];
        3*x[1] - x[2] - 3 - x[6];
        -x[1] + 0.5*x[2] + 4 - x[7];
        -x[1] - x[2] + 7 - x[8];
    ]

    inequality_constraints(x,θ) = [
        x;
    ] 

    complementarity_constraints(x,θ) = [-(x[3]*x[6] + x[4]*x[7] + x[5]*x[8])]
    primal_dimension = 8


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
        relaxation_mode = :standard,
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