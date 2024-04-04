struct ParametricMPCC{T1<:Vector{<:ParametricOptimizationProblem}}
    subproblems::T1
    complementarity_dimension::Int
end

"""
Synthesize a parametric MPCC problem 
"""

function ParametricMPCC(;
    objective,
    equality_constraints,
    inequality_constraints,
    complementarity_constraints, # Φ(Gᵢ(x), Hᵢ(x)) ≤ 0, i = 1,...,m
    primal_dimension,   
    parameter_dimension,
    relaxation_parameter,
    update_parameter,
    max_iterations,
    relaxation_mode = :standard,
    )

    @assert 0.0 < update_parameter < 1.0 "Update parameter must be in the range (0,1)"

    # Problem data
    dummy_parameters = zeros(parameter_dimension)

    subproblems = (ParametricOptimizationProblem[])

    # First subproblem is the original MPCC
    relaxations = vcat([0], relaxation_parameter * update_parameter.^(0:max_iterations))
    
    # Set relaxation mode
    if relaxation_mode === :standard
        dummy_primals = zeros(primal_dimension)

        equality_dimension = length(equality_constraints(dummy_primals, dummy_parameters))
        complementarity_dimension = length(complementarity_constraints(dummy_primals, dummy_parameters))
        inequality_dimension = length(inequality_constraints(dummy_primals, dummy_parameters)) + complementarity_dimension

        objective_ϵ = objective
        for ϵ in relaxations
            combined_inequality_constraints = function (x,θ)
                [inequality_constraints(x,θ); complementarity_constraints(x,θ) .+ ϵ]
            end

            relaxed_problem = ParametricOptimizationProblem(;
                objective = objective_ϵ,
                equality_constraint = equality_constraints,
                inequality_constraint = combined_inequality_constraints,
                parameter_dimension = parameter_dimension,
                primal_dimension = primal_dimension,   
                equality_dimension = equality_dimension,
                inequality_dimension = inequality_dimension,
            )

            push!(subproblems, relaxed_problem)
        end

    elseif relaxation_mode === :l_infinity
        primal_dimension = primal_dimension + 1
        dummy_primals = zeros(primal_dimension)

        equality_dimension = length(equality_constraints(dummy_primals, dummy_parameters))
        complementarity_dimension = length(complementarity_constraints(dummy_primals, dummy_parameters))
        inequality_dimension = length(inequality_constraints(dummy_primals, dummy_parameters)) + complementarity_dimension

        for ϵ in relaxations
            if isequal(ϵ, 0.0)
                objective_ϵ = objective
                combined_inequality_constraints = function (x,θ)
                    [inequality_constraints(x,θ); complementarity_constraints(x,θ)]
                end
            else
                objective_ϵ = function (x,θ)
                    objective(x,θ) +  x[primal_dimension] / ϵ
                end
                combined_inequality_constraints = function (x,θ)
                    [inequality_constraints(x,θ); complementarity_constraints(x,θ) .+ x[primal_dimension]]
                end
            end
            inequality_dimension = length(combined_inequality_constraints(dummy_primals, dummy_parameters))
            relaxed_problem = ParametricOptimizationProblem(;
                objective = objective_ϵ,
                equality_constraint = equality_constraints,
                inequality_constraint = combined_inequality_constraints,
                parameter_dimension = parameter_dimension,
                primal_dimension = primal_dimension,   
                equality_dimension = equality_dimension,
                inequality_dimension = inequality_dimension,
            )

            push!(subproblems, relaxed_problem)
        end
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

    if isnothing(initial_guess)
        initial_guess = zeros(total_dim(original_problem))
    end

    # Last "m" inequality constraints are complementarity constraints
    complementarity_dimension = problem.complementarity_dimension
    complementarity_residual = 1.0

    # Start from the second subproblem: First subproblem is the original MPCC
    ii = 2 
    while complementarity_residual > tolerance && ii ≤ max_iterations + 2

        # If the relaxed problem is infeasible, terminate. Otherwise solve the relaxed problem for xᵏ⁺¹
        relaxed_problem = problem.subproblems[ii]
        solution = solve(relaxed_problem, parameters; initial_guess)
        if verbose
            println("ii: ", ii-1)
            println("status: ", solution.status)
            println("primals: ", solution.primals)
            println("objective: ", original_problem.objective(solution.primals, parameters))
        end

        if string(solution.status) != "MCP_Solved" 
            verbose && printstyled("Could not solve relaxed problem at relaxation factor $(ii-1).\n"; color = :red)
            break
        end

        # Check complementarity residual
        complementarity_violations = last(original_problem.inequality_constraint(solution.primals, parameters), complementarity_dimension)
        complementarity_residual = findmax(-complementarity_violations)[1]

        # Stop if iteration does not improve. Otherwise update initial_guess
        if norm(initial_guess - solution.variables) < tolerance
            verbose && printstyled("Converged at iteration $(ii-1).\n"; color = :green)
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

