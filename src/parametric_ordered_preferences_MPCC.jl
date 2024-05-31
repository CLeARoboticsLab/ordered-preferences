struct ParametricOrderedPreferencesMPCC{T1<:ParametricOptimizationProblem, T2, T3}
    relaxed_problem::T1
    exact_complementarity_constraints::T2
    objective::T3
end

"""
Synthesizes a parametric ordered preferences problem (priority_levels >= 2) as an MPCC 
(an optimization problem with single objective and KKT constraints resulting from inner levels)
Inputs: 
    ...
Outputs: 
    ...
"""

function ParametricOrderedPreferencesMPCC(;
    objective,
    equality_constraints,
    inequality_constraints,
    prioritized_preferences,
    is_prioritized_constraint, #TODO: Better way of reconciling between prioritized obj and constraints?
    primal_dimension,
    parameter_dimension,
    equality_dimension,
    inequality_dimension,
    relaxation_mode = :standard,
) 
    # Problem data
    ordered_priority_levels = eachindex(prioritized_preferences)

    dual_dimension = 0

    inner_equality_constraints = Function[equality_constraints]
    inner_inequality_constraints = Function[inequality_constraints]
    inner_complementarity_constraints = Function[]
    final_complementarity_constraint = Function[]

    inequality_dimension_ii = inequality_dimension
    equality_dimension_ii = equality_dimension
    original_primal_dimension = primal_dimension
    original_parameter_dimension = parameter_dimension

    # one extra parameter for relaxation
    augmented_parameter_dimension = parameter_dimension + 1

    dummy_primals = zeros(primal_dimension)
    dummy_parameters = zeros(augmented_parameter_dimension)

    function set_up_level(priority_level)
        # Implement prioritized objective vs prioritized constraints
        if is_prioritized_constraint[priority_level]
            prioritized_constraints_ii = prioritized_preferences[priority_level] # fᵢ(x,θ) ≥ 0 

            slack_dimension_ii = length(prioritized_constraints_ii(dummy_primals, dummy_parameters))
            primal_dimension = primal_dimension + slack_dimension_ii
            inequality_dimension_ii = inequality_dimension_ii + slack_dimension_ii
        end
        
        primal_dimension_ii = primal_dimension + dual_dimension

        # Define symbolic variables for primals.
        total_dimension = primal_dimension_ii + inequality_dimension_ii + equality_dimension_ii
        z̃ = Symbolics.scalarize(only(Symbolics.@variables(z̃[1:total_dimension])))
        z = BlockArray(z̃, [primal_dimension_ii, inequality_dimension_ii, equality_dimension_ii])

        x = z[Block(1)]
        λ = z[Block(2)]
        μ = z[Block(3)]

        # Define symbolic variables for (augmented) parameters.
        θ̃ = only(Symbolics.@variables(θ̃[1:augmented_parameter_dimension]))
        θ = Symbolics.scalarize(θ̃)
        if isempty(θ)
            θ = Symbolics.Num[]
        end

        # Build symbolic expression for objective and constraints. 
        if is_prioritized_constraint[priority_level]

            slacks_ii = last(x, slack_dimension_ii)

            # objective: minimize sum of squared slacks, min ∑sᵢ²
            slack_objective = function (x,θ)
                sum(last(x, slack_dimension_ii) .^ 2)
            end
            objective_ii = slack_objective(x,θ)

            # auxillary constraint: fᵢ(x,θ) + sᵢ ≥ 0 (sᵢ ≥ 0 is implicit)
            auxillary_constraints = function(x,θ)
                original_x = x[1:original_primal_dimension]
                original_θ = θ[1:original_parameter_dimension]
                prioritized_constraints_ii(original_x, original_θ) .+ slacks_ii
            end
            push!(inner_inequality_constraints, auxillary_constraints)
        else
            priority_objective_ii = prioritized_preferences[priority_level]
            objective_ii = priority_objective_ii(x,θ)
        end

        h_ii = mapreduce(vcat, inner_inequality_constraints) do constraint
            constraint(x,θ)
        end
        if isempty(h_ii)
            h_ii = Symbolics.Num[]
        end

        g_ii = mapreduce(vcat, inner_equality_constraints) do constraint
            constraint(x,θ)
        end
        if isempty(g_ii)
            g_ii = Symbolics.Num[]
        end

        # Lagrangian.
        L = objective_ii - λ' * h_ii + μ' * g_ii 

        # Stationary constraints (# ∇ₓL = 0).
        stationarity = Symbolics.gradient(L, x) 

        # Concatenate stationary constraint into equality constraints.
        callable_stationarity = Symbolics.build_function(stationarity, z̃, θ,  expression=Val{false})[1]
        push!(inner_equality_constraints, callable_stationarity)

        # Dual nonnegativity constraints
        dual_nonnegativity = λ

        # Concatenate dual_nonnegativity constraints into inequality constraints.
        callable_dual_nonnegativity = Symbolics.build_function(dual_nonnegativity, z̃, θ, expression=Val{false})[1]
        push!(inner_inequality_constraints, callable_dual_nonnegativity)

        # Complementarity constraints from inequality constraints.
        complementarity = -h_ii .* λ

        # Reduce relaxed (inner) complementarity_constraints into a single inequality and concatenate.
        if priority_level < last(ordered_priority_levels)
            for problem in (:relaxed, :exact)
                if problem === :relaxed 
                    # The last parameter is the relaxation parameter
                    relaxed_complementarity = sum(complementarity) .+ θ[augmented_parameter_dimension] 
                    callable_complementarity = Symbolics.build_function(relaxed_complementarity, z̃, θ, expression=Val{false})
                    push!(inner_inequality_constraints, callable_complementarity)
                else
                    # Exact problem has no relaxation.
                    callable_complementarity = Symbolics.build_function(complementarity, z̃, θ, expression=Val{false})[1]
                    push!(inner_complementarity_constraints, callable_complementarity)
                end
            end
        else 
            # Final level: relaxation happens in MPCC 
            relaxed_complementarity = complementarity
            final_complementarity = Symbolics.build_function(relaxed_complementarity, z̃, θ, expression=Val{false})[1]
            push!(final_complementarity_constraint, final_complementarity)
        end

        # Update dual dimension, inequality and equality dimension for the next level. 
        dual_dimension += inequality_dimension_ii + equality_dimension_ii
        inequality_dimension_ii += length(dual_nonnegativity) + length(relaxed_complementarity)
        equality_dimension_ii += length(stationarity)
    end

    # Build KKT system for each priority level
    for priority_level in ordered_priority_levels
        set_up_level(priority_level)
    end
    
    # Problem setting for final/outermost level 
    primal_dimension = primal_dimension + dual_dimension
    dummy_primals = zeros(primal_dimension)

    equality_constraints = function(x,θ)
        mapreduce(vcat, inner_equality_constraints) do constraint
            constraint(x,θ)
        end
    end
    equality_dimension = length(equality_constraints(dummy_primals, dummy_parameters))
    @assert equality_dimension == equality_dimension_ii

    inequality_constraints = function(x,θ)
        mapreduce(vcat, inner_inequality_constraints) do constraint
            constraint(x,θ)
        end
    end
    exact_final_complementarity = function (x,θ)
        final_complementarity_constraint[1](x,θ)
    end
    inequality_dimension = length(inequality_constraints(dummy_primals, dummy_parameters)) + length(exact_final_complementarity(dummy_primals, dummy_parameters))
    @assert inequality_dimension == inequality_dimension_ii 

    # Use the exact complementarity constraints to evaluate convergence
    if !isempty(inner_complementarity_constraints)
        exact_complementarity_constraints = function (x,θ)
        [
            exact_final_complementarity(x,θ); 
            mapreduce(vcat, inner_complementarity_constraints) do constraint
                constraint(x,θ)
            end
        ]
        end
    else
        # Prob has only one priority level
        exact_complementarity_constraints = function (x,θ) exact_final_complementarity(x,θ) end
    end

    if relaxation_mode === :standard
        objective_ϵ = objective

        combined_inequality_constraints = function (x,θ) 
            [
                inequality_constraints(x,θ); 
                exact_final_complementarity(x,θ) .+ θ[augmented_parameter_dimension]
            ]
        end

    elseif relaxation_mode === :l_infinity
        primal_dimension = primal_dimension + 1

        objective_ϵ = function (x,θ)  
            objective(x,θ) + x[primal_dimension] ./ θ[augmented_parameter_dimension]
        end
        combined_inequality_constraints = function (x,θ)
            [
                inequality_constraints(x,θ); 
                exact_final_complementarity(x,θ) .+ x[primal_dimension]
            ]
        end

    else
        error("Unknown relaxation mode: $relaxation_mode")
    end
    
    relaxed_problem = ParametricOptimizationProblem(;
        objective = objective_ϵ, 
        equality_constraint = equality_constraints,
        inequality_constraint = combined_inequality_constraints,
        parameter_dimension = augmented_parameter_dimension,
        primal_dimension,
        equality_dimension,
        inequality_dimension,
    )    
    ParametricOrderedPreferencesMPCC(relaxed_problem, exact_complementarity_constraints, objective)
end

function solve_relaxed_pop(
    problem::ParametricOrderedPreferencesMPCC,
    initial_guess::Union{Nothing, Vector{Float64}},
    parameters;
    ϵ = 1.0,
    κ = 0.1,
    max_iterations = 10,
    tolerance = 1e-7,
    verbose = false
)
    solutions = []

    relaxed_problem = problem.relaxed_problem
    exact_complementarity_constraints = problem.exact_complementarity_constraints
    original_objective = problem.objective

    if isnothing(initial_guess)
        initial_guess = zeros(total_dim(relaxed_problem))
    end

    complementarity_residual = 1.0
    converged_tolerance = 1e-20

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
            println("objective: ", original_objective(solution.primals, augmented_parameters)) 
        end

        if string(solution.status) != "MCP_Solved" 
            verbose && printstyled("Could not solve relaxed problem at relaxation factor $(ii).\n"; color = :red)
            break
        end
        # Check complementarity residual
        complementarity_violations = exact_complementarity_constraints(solution.primals, augmented_parameters)
        complementarity_residual = findmax(-complementarity_violations)[1]

        # # Stop if iteration does not improve. 
        # if norm(initial_guess - solution.variables) < converged_tolerance 
        #     verbose && printstyled("Converged at iteration $(ii).\n"; color = :green)
        #     break
        # end

        # Update initial_guess
        initial_guess = solution.variables 
        push!(solutions, solution)
        
        # Begin next iteration
        ϵ = κ * ϵ
        ii += 1
    end

    verbose && if complementarity_residual < tolerance
        printstyled("Found a solution with complementarity residual less than tol=$(tolerance).\n"; color = :blue)
    end

    verbose && println("complementarity_residual: ", complementarity_residual)
    (; relaxation = ϵ, solution = solutions, residual = complementarity_residual) #TODO solutions[end]
end 