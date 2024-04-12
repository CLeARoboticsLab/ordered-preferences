struct ParametricMPCC{T1}
    "The MPCC reformulated as regular optimization problem with an additional runtime parameter for relaxation"
    relaxed_parametric_optimization_problem::T1
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
    relaxation_parameter_schedule = [1.0, 0.1, 0.01, 0.001],
    update_parameter,
    max_iterations,
    relaxation_mode = :standard,
)
    augmented_parameter_dimension = parameter_dimension + 1 # +1 for relaxation parameter

    @assert 0.0 < update_parameter < 1.0 "Update parameter must be in the range (0,1)"

    # Problem data
    dummy_parameters = zeros(parameter_dimension)

    # Set relaxation mode
    #    if relaxation_mode === :standard
    dummy_primals = zeros(primal_dimension)

    equality_dimension = length(equality_constraints(dummy_primals, dummy_parameters))
    complementarity_dimension =
        length(complementarity_constraints.H(dummy_primals, dummy_parameters))
    @assert length(complementarity_constraints.G(dummy_primals, dummy_parameters)) ==
            complementarity_dimension

    augmented_inequality_dimension =
        length(inequality_constraints(dummy_primals, dummy_parameters)) +
        3 * complementarity_dimension # additional constraints for G(x) >= 0, H(x) >= 0, G(x) .* H(x) <= ϵ

    # Same scheduling of relaxation parameters for all levels
    combined_inequality_constraints = function (x, θ_augmented)
        θ = θ_augmented[1:parameter_dimension]
        ϵ = θ_augmented[parameter_dimension + 1]

        Gxθ = complementarity_constraints.G(x, θ)
        Hxθ = complementarity_constraints.H(x, θ)
        [
            inequality_constraints(x, θ)
            Gxθ
            Hxθ
            -Gxθ .* Hxθ .+ ϵ
        ]
    end

    relaxed_parametric_optimization_problem = ParametricOptimizationProblem(;
        objective,
        equality_constraint = equality_constraints,
        inequality_constraint = combined_inequality_constraints,
        parameter_dimension = augmented_parameter_dimension,
        primal_dimension = primal_dimension,
        equality_dimension = equality_dimension,
        inequality_dimension = augmented_inequality_dimension,
    )
    #    elseif relaxation_mode === :l_infinity
    #        primal_dimension = primal_dimension + 1
    #        dummy_primals = zeros(primal_dimension)
    #
    #        equality_dimension = length(equality_constraints(dummy_primals, dummy_parameters))
    #        complementarity_dimension =
    #            length(complementarity_constraints(dummy_primals, dummy_parameters))
    #        inequality_dimension =
    #            length(inequality_constraints(dummy_primals, dummy_parameters)) +
    #            complementarity_dimension
    #
    #        for ϵ in relaxations
    #            ϵ_inner = ϵ * ones(parameter_dimension)
    #            if isequal(ϵ, 0.0)
    #                objective_ϵ = objective
    #                combined_inequality_constraints = function (x, θ)
    #                    [inequality_constraints(x, θ); complementarity_constraints(x, θ)]
    #                end
    #            else
    #                objective_ϵ = function (x, θ)
    #                    objective(x, θ) + x[primal_dimension] / ϵ
    #                end
    #                combined_inequality_constraints = function (x, ϵ_inner)
    #                    [
    #                        inequality_constraints(x, ϵ_inner)
    #                        complementarity_constraints(x, ϵ_inner) .+ x[primal_dimension]
    #                    ]
    #                end
    #            end
    #            inequality_dimension =
    #                length(combined_inequality_constraints(dummy_primals, dummy_parameters))
    #            relaxed_problem = ParametricOptimizationProblem(;
    #                objective = objective_ϵ,
    #                equality_constraint = equality_constraints,
    #                inequality_constraint = combined_inequality_constraints,
    #                parameter_dimension = parameter_dimension,
    #                primal_dimension = primal_dimension,
    #                equality_dimension = equality_dimension,
    #                inequality_dimension = inequality_dimension,
    #            )
    #
    #            push!(subproblems, relaxed_problem)
    #        end
    #    else
    #        error("Invalid relaxation mode")
    #    end

    ParametricMPCC(subproblems, complementarity_dimension)
end

"""
- 'problem' is a group of relaxed MPCCs
"""

function solve_relaxed_mpcc(
    problem::ParametricMPCC,
    initial_guess::Union{Nothing,Vector{Float64}},
    parameters;
    ϵ = 1.0,
    κ = 0.1,
    max_iterations = 10,
    tolerance = 1e-7,
    verbose = false,
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
            println("ii: ", ii - 1)
            println("status: ", solution.status)
            println("primals: ", solution.primals)
            println("objective: ", original_problem.objective(solution.primals, parameters))
        end

        if string(solution.status) != "MCP_Solved"
            verbose && printstyled(
                "Could not solve relaxed problem at relaxation factor $(ii-1).\n";
                color = :red,
            )
            break
        end

        # Check complementarity residual
        complementarity_violations = last(
            original_problem.inequality_constraint(solution.primals, parameters),
            complementarity_dimension,
        )
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
