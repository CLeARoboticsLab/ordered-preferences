"""
Synthesizes a parametric ordered preferences problem from user functions
"""
function build_ordered_preferences_problem(;
    objective,
    equality_constraints,
    inequality_constraints,
    prioritized_inequality_constraints,
    primal_dimension,
    parameter_dimension,
)
    # Problem data
    ordered_priority_levels = sort(collect(keys(prioritized_inequality_constraints)); rev = true)
    outer_level = ordered_priority_levels[end]

    dummy_primals = zeros(primal_dimension)
    dummy_parameters = zeros(parameter_dimension)

    equality_dimension = length(equality_constraints(dummy_primals, dummy_parameters))
    inequality_dimension =
        length(prioritized_inequality_constraints[outer_level](dummy_primals, dummy_parameters))

    total_inner_slack_dimension = 0
    inner_inequality_constraints = Any[inequality_constraints]

    ordered_preferences_problem = ParametricOptimizationProblem[]

    function set_up_level(priority_level)
        parameter_dimension_ii = parameter_dimension + total_inner_slack_dimension

        if isnothing(priority_level)
            # the final level does not have any additional slacks
            slack_dimension_ii = 0
        else
            prioritized_constraints_ii = prioritized_inequality_constraints[priority_level]
            slack_dimension_ii = length(prioritized_constraints_ii(dummy_primals, dummy_parameters))
        end

        primal_dimension_ii = primal_dimension + slack_dimension_ii

        if isnothing(priority_level)
            objective_ii = objective
        else
            objective_ii = function (x, θ)
                # everything beyond the original primal dimension are the slacks for this level
                sum(x[(primal_dimension + 1):end] .^ 2)
            end
        end

        inequality_constraints_ii = function (x, θ)
            original_θ = θ[1:parameter_dimension]
            fixed_slacks = θ[(parameter_dimension + 1):end]
            @assert length(fixed_slacks) == total_inner_slack_dimension
            slacks_ii = x[(primal_dimension + 1):end]
            @assert length(slacks_ii) == slack_dimension_ii

            unslacked_constraints =
                mapreduce(vcat, inner_inequality_constraints) do constraint
                    constraint(x, original_θ)
                end + vcat(zeros(inequality_dimension), fixed_slacks)

            if isnothing(priority_level)
                return unslacked_constraints
            end

            vcat(unslacked_constraints, prioritized_constraints_ii(x, original_θ) .+ slacks_ii)
        end

        inequality_dimension_ii = let
            dummy_primals_ii = zeros(primal_dimension_ii)
            dummy_parameter_ii = zeros(parameter_dimension_ii)
            length(inequality_constraints_ii(dummy_primals_ii, dummy_parameter_ii))
        end

        optimization_problem = ParametricOptimizationProblem(;
            objective = objective_ii,
            equality_constraint = equality_constraints,
            inequality_constraint = inequality_constraints_ii,
            parameter_dimension = parameter_dimension_ii,
            primal_dimension = primal_dimension_ii,
            equality_dimension = equality_dimension,
            inequality_dimension = inequality_dimension_ii,
        )

        total_inner_slack_dimension += slack_dimension_ii
        if !isnothing(priority_level)
            push!(inner_inequality_constraints, prioritized_inequality_constraints[priority_level])
        end
        push!(ordered_preferences_problem, optimization_problem)
    end

    for priority_level in ordered_priority_levels
        set_up_level(priority_level)
    end
    set_up_level(nothing)

    ordered_preferences_problem
end
