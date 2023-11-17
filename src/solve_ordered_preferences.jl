# TODO: for now `ordered_preferences_problem` is a vector of ParametricOptimizationProblem,
# could introduce a struct for that
# TODO: allow for user-defined warm-starting
function solve_ordered_preferences(ordered_preferences_problem, θ)
    outer_problem = last(ordered_preferences_problem)

    # Initial guess:
    fixed_slacks = Float64[]

    # TODO: optimize this
    inner_solution = nothing
    for optimization_problem in ordered_preferences_problem
        initial_guess = zeros(total_dim(optimization_problem))
        if !isnothing(inner_solution)
            initial_guess[1:(optimization_problem.primal_dimension)] =
                inner_solution.primals[1:(optimization_problem.primal_dimension)]
        end

        parameter_value = vcat(θ, fixed_slacks)
        solution = solve(optimization_problem, parameter_value; initial_guess)
        append!(fixed_slacks, solution.primals[(outer_problem.primal_dimension + 1):end])
        inner_solution = solution
    end

    inner_solution
end
