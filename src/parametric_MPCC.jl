struct ParametricMPCC{T1<:Vector{<:ParametricOptimizationProblem}}
    subproblems::T1
    complementarity_dimension::Int
end

"""
Synthesize a parametric MPCC problem (suitable for bi-level programming)
"""

function ParametricMPCC(;
    objective,
    equality_constraints,
    inequality_constraints,
    complementarity_constraints, # Φ(Gᵢ(x), Hᵢ(x)) ≤ 0, i = 1,...,m
    primal_dimension,   
    parameter_dimension,
    relaxation_mode = :standard,
    )

    # Problem data
    dummy_parameters = zeros(parameter_dimension)

    subproblems = (ParametricOptimizationProblem[])
    
    # Set relaxation mode
    if relaxation_mode === :standard
        dummy_primals = zeros(primal_dimension)

        equality_dimension = length(equality_constraints(dummy_primals, dummy_parameters))
        complementarity_dimension = length(complementarity_constraints(dummy_primals, dummy_parameters))
        inequality_dimension = length(inequality_constraints(dummy_primals, dummy_parameters)) + complementarity_dimension

        for problem in (:original, :relaxed)
            if problem === :original
                combined_inequality_constraints = function (x,θ) 
                    [inequality_constraints(x,θ); complementarity_constraints(x,θ)]
                end
            else
                # Same scheduling of relaxation parameters for all levels 
                combined_inequality_constraints = function (x,θ) 
                    [inequality_constraints(x,θ); complementarity_constraints(x,θ) .+ θ[parameter_dimension]]
                end
            end
            parametric_optimization_problem = ParametricOptimizationProblem(;
                objective = objective,
                equality_constraint = equality_constraints,
                inequality_constraint = combined_inequality_constraints,
                parameter_dimension,  # = augmented_parameter_dimension
                primal_dimension,   
                equality_dimension,
                inequality_dimension,
            )
            push!(subproblems, parametric_optimization_problem)
        end

    # elseif relaxation_mode === :l_infinity
    #     primal_dimension = primal_dimension + 1
    #     dummy_primals = zeros(primal_dimension)

    #     equality_dimension = length(equality_constraints(dummy_primals, dummy_parameters))
    #     complementarity_dimension = length(complementarity_constraints(dummy_primals, dummy_parameters))
    #     inequality_dimension = length(inequality_constraints(dummy_primals, dummy_parameters)) + complementarity_dimension

    #     for ϵ in relaxations # TODO
    #         if isequal(ϵ, 0.0)
    #             objective_ϵ = objective
    #             combined_inequality_constraints = function (x,θ)
    #                 [inequality_constraints(x,θ); complementarity_constraints(x,θ)]
    #             end
    #         else
    #             objective_ϵ = function (x,θ)  
    #                 objective(x,θ) +  x[primal_dimension] / ϵ
    #             end
    #             combined_inequality_constraints = function (x,θ)
    #                 [inequality_constraints(x,θ); complementarity_constraints(x,θ) .+ x[primal_dimension]]
    #             end
    #         end
    #         inequality_dimension = length(combined_inequality_constraints(dummy_primals, dummy_parameters))
    #         relaxed_problem = ParametricOptimizationProblem(;
    #             objective = objective_ϵ,
    #             equality_constraint = equality_constraints,
    #             inequality_constraint = combined_inequality_constraints,
    #             parameter_dimension = parameter_dimension,
    #             primal_dimension = primal_dimension,   
    #             equality_dimension = equality_dimension,
    #             inequality_dimension = inequality_dimension,
    #         )

    #     end
    else
        error("Invalid relaxation mode")
    end

    ParametricMPCC(subproblems, complementarity_dimension)
end

"""
- 'problem' is a group of relaxed MPCCs
"""

function solve_relaxed_mpcc(
    problem::ParametricMPCC,
    initial_guess::Union{Nothing, Vector{Float64}},
    parameters;
    ϵ = 1.0,
    κ = 0.1,
    max_iterations = 10,
    tolerance = 1e-7,
    verbose = false
)
    solutions = []

    original_problem = problem.subproblems[1]
    relaxed_problem = problem.subproblems[2]

    if isnothing(initial_guess)
        initial_guess = zeros(total_dim(original_problem))
    end

    # Last "m" inequality constraints are complementarity constraints
    complementarity_dimension = problem.complementarity_dimension
    complementarity_residual = 1.0

    relaxations = ϵ * κ.^(0:max_iterations) # [1.0, 0.1, 0.01, ... 1e-10]
    ii = 1 
    while complementarity_residual > tolerance && ii ≤ max_iterations + 1
        # If the relaxed problem is infeasible, terminate. Otherwise solve the relaxed problem for xᵏ⁺¹
        ϵ = relaxations[ii]
        augmented_parameters = vcat(parameters, ϵ)
        solution = solve(relaxed_problem, augmented_parameters; initial_guess)
        if verbose
            println("ii: ", ii)
            println("status: ", solution.status)
            println("primals: ", solution.primals)
            println("objective: ", original_problem.objective(solution.primals, augmented_parameters))
        end

        if string(solution.status) != "MCP_Solved" 
            verbose && printstyled("Could not solve relaxed problem at relaxation factor $(ii-1).\n"; color = :red)
            break
        end
        # Check complementarity residual
        complementarity_violations = last(original_problem.inequality_constraint(solution.primals, augmented_parameters), complementarity_dimension)
        complementarity_residual = findmax(-complementarity_violations)[1]

        # Stop if iteration does not improve. Otherwise update initial_guess
        if norm(initial_guess - solution.variables) < tolerance
            verbose && printstyled("Converged at iteration $(ii).\n"; color = :green)
            break
        else
            initial_guess = solution.variables 
            push!(solutions, solution)
        end

        # Begin next iteration
        ϵ = κ * ϵ
        ii += 1
    end

    verbose && println("complementarity_residual: ", complementarity_residual)
    (; relaxation = ϵ, solution = solutions[end], residual = complementarity_residual)
end 

