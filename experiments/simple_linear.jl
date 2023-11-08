using OrderedPreferences
using BlockArrays: Block
using LinearAlgebra

"""
Infeasible LP 
min₍ₓ₁,ₓ₂₎  x₁ + x₂
s.t.        g₁(x₁, x₂) = x₁ ≥ 6,
            g₂(x₁, x₂) = x₂ ≥ 6
            g₃(x₁, x₂) = x₁ + x₂ ≤ 11
Assume g₃ more important than g₂, i.e. innermost problem minimizes slack for g₃.
"""
# TODO  
    #g(x,θ) = [g(x,θ); gₐ(x,θ)] # updated/merged inequality constraint 
    #try_g(x,θ) = [g(x,θ)[1]; g(x,θ)[2]]

function build_parametric_optim(problem::ParametricOptimizationProblem, preference)

    # Count number of constraints with preferences
    count_preference = count(x -> x !=0 , preference) 

    # Indices of constraints without preferences
    no_preference = problem.inequality_dimension - count_preference

    # Problem data
    parameter_dimension = problem.parameter_dimension
    primal_dimension = problem.primal_dimension
    equality_dimension = problem.equality_dimension

    # Start with innermost problem
    J(x, θ) = x[primal_dimension + 1] 
    f(x,θ) = problem.equality_constraint(x,θ)
    g(x,θ) = vcat(problem.inequality_constraint(x,θ)[1:no_preference], 
            [problem.inequality_constraint(x,θ)[end] + x[primal_dimension + 1],
            x[primal_dimension + 1]]
            )     

    inner_problem = ParametricOptimizationProblem(;
        objective = J, 
        equality_constraint = f,
        inequality_constraint = g,
        parameter_dimension = parameter_dimension,
        primal_dimension = primal_dimension + 1,  
        equality_dimension = equality_dimension, 
        inequality_dimension = no_preference + 2, 
        )
    
    # Initial guess: 
    z₀ = zeros(total_dim(inner_problem)) 

    (; primals, variables, status, info) = solve(inner_problem, [0]; initial_guess = z₀)
    println("Level: ", count_preference)
    println("primals: ", primals)
    println("variables: ", variables)
    println("status: ", status)
    println("info: ", info)

    # Loop over intermediate levels
    for ii ∈ count_preference-1:-1:1
        
        Jᵢ(x,θ) = x[inner_problem.primal_dimension + 1]
        fᵢ(x,θ) = problem.equality_constraint(x,θ)
        gᵢ(x,θ) = vcat(inner_problem.inequality_constraint(x,θ), 
                [problem.inequality_constraint(x,θ)[ii + no_preference] + x[inner_problem.primal_dimension + 1],
                x[inner_problem.primal_dimension + 1]]
                ) 

        inner_problem = ParametricOptimizationProblem(;
            objective = Jᵢ, 
            equality_constraint = fᵢ,
            inequality_constraint = gᵢ,
            parameter_dimension = parameter_dimension,
            primal_dimension = inner_problem.primal_dimension + 1, 
            equality_dimension = equality_dimension, 
            inequality_dimension = inner_problem.inequality_dimension + 2, 
            )

        #Main.@infiltrate

        # Initial guess 
        z₀ = vcat(primals, zeros(total_dim(inner_problem) - inner_problem.primal_dimension + 1))

        (; primals, variables, status, info) = solve(inner_problem, [0]; initial_guess = z₀)
        println("Level: ", ii)
        println("primals: ", primals)
        println("variables: ", variables)
        println("status: ", status)
        println("info: ", info)
    end

    (; primals, variables, status, info)
end

function simple_linear()

    # Define Original (infeasible) Problem
    J₀(x,θ) = sum(x)
    f(x,θ) = [0] 
    g(x,θ) = [x[1] - 6.0, 
              x[2] - 6.0, 
             -x[1] - x[2] + 11.0,
            ]
    
    # Define preferences for constraint ordering  #[0,...,0,1,2,...,n]
    preference = [0, 1, 2]

    problem = ParametricOptimizationProblem(;
        objective = J₀, 
        equality_constraint = f,
        inequality_constraint = g,
        parameter_dimension = 1,
        primal_dimension = 2, 
        equality_dimension = 1, 
        inequality_dimension = 3,
        )
    
    (; primals, variables, status, info) = build_parametric_optim(problem, preference)

    # TODO: Solve the relaxed version of original problem 
    

end 


