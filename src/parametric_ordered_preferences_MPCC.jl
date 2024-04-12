# struct ParametricOrderedPreferencesMPCC{T1<:Vector{<:ParametricOptimizationProblem}}
#     subproblems::T1
# end

"""
Synthesizes a parametric ordered preferences problem as an MPCC 
(an optimization problem with single objective and KKT constraints resulting from inner levels)
"""

function ParametricOrderedPreferencesMPCC(;
    objective,
    equality_constraints,
    inequality_constraints,
    prioritized_preferences,
    primal_dimension,
    parameter_dimension,
    equality_dimension,
    inequality_dimension,
    relaxation_parameter,
    update_parameter,
    max_iterations,
    relaxation_mode = :standard,
)
    # Problem data
    ordered_priority_levels = eachindex(prioritized_preferences)

    dual_dimension = 0

    inner_equality_constraints = Function[equality_constraints]
    inner_inequality_constraints = Function[inequality_constraints]
    complementarity_constraints = Function[]

    inequality_dimension_ii = inequality_dimension
    equality_dimension_ii = equality_dimension

    function set_up_level(priority_level)
        is_inner_most_problem = priority_level == 1
        if is_inner_most_problem
            augmented_parameter_dimension = parameter_dimension
        else
            augmented_parameter_dimension = parameter_dimension + 1 # one extra parameter for relaxation
        end

        #TODO: Implement priority objective (for now) vs priority constraints 
        primal_dimension_ii = primal_dimension + dual_dimension
        priority_preferences_ii = prioritized_preferences[priority_level]

        # Define symbolic variables for primals.
        total_dimension = primal_dimension_ii + inequality_dimension_ii + equality_dimension_ii
        z̃ = Symbolics.scalarize(only(Symbolics.@variables(z̃[1:total_dimension])))
        z = BlockArray(z̃, [primal_dimension_ii, inequality_dimension_ii, equality_dimension_ii])

        x = z[Block(1)]
        λ = z[Block(2)]
        μ = z[Block(3)]

        # Define symbolic variables for parameters.
        θ̃ = only(Symbolics.@variables(θ̃[1:parameter_dimension]))
        θ = Symbolics.scalarize(θ̃)
        if isempty(θ)
            θ = Symbolics.Num[]
        end

        # Build symbolic expression for objective and constraints. 
        objective_ii = priority_preferences_ii(x, θ)

        g_ii = mapreduce(vcat, inner_equality_constraints) do constraint
            constraint(x, θ)
        end
        if isempty(g_ii)
            g_ii = Symbolics.Num[]
        end

        h_ii = mapreduce(vcat, inner_inequality_constraints) do constraint
            constraint(x, θ)
        end
        if isempty(h_ii)
            h_ii = Symbolics.Num[]
        end

        # Lagrangian.
        L = objective_ii - λ' * h_ii + μ' * g_ii

        # Stationary constraints (# ∇ₓL = 0).
        stationarity = Symbolics.gradient(L, x)

        # Concatenate stationary constraint into equality constraints.
        callable_stationarity =
            Symbolics.build_function(stationarity, z̃, θ, expression = Val{false})[1]
        push!(inner_equality_constraints, callable_stationarity)

        # Dual nonnegativity constraints
        dual_nonnegativity = λ

        # Concatenate dual_nonnegativity constraints into inequality constraints.
        callable_dual_nonnegativity =
            Symbolics.build_function(dual_nonnegativity, z̃, θ, expression = Val{false})[1]
        push!(inner_inequality_constraints, callable_dual_nonnegativity)

        # Complementarity constraints from inequality constraints.
        complementarity = -h_ii .* λ

        # Reduce relaxed complementarity_constraints into a single inequality and concatenate.
        if priority_level < last(ordered_priority_levels)
            relaxed_complementarity = sum(complementarity) + θ[priority_level] #TODO: Is this right?
            callable_complementarity =
                Symbolics.build_function(relaxed_complementarity, z̃, θ, expression = Val{false})
            push!(inner_inequality_constraints, callable_complementarity)
        else
            # Final level: relaxation happens in MPCC 
            relaxed_complementarity = complementarity
            callable_complementarity =
                Symbolics.build_function(relaxed_complementarity, z̃, θ, expression = Val{false})[1]
            push!(complementarity_constraints, callable_complementarity)
        end

        # Update dual dimension, inequality and equality dimension for the next level. 
        dual_dimension += inequality_dimension_ii + equality_dimension_ii
        inequality_dimension_ii += length(dual_nonnegativity) + length(relaxed_complementarity)
        equality_dimension_ii += length(stationarity)
    end

    for priority_level in ordered_priority_levels
        set_up_level(priority_level)
    end

    primal_dimension = primal_dimension + dual_dimension

    equality_constraints = function (x, θ)
        mapreduce(vcat, inner_equality_constraints) do constraint
            constraint(x, θ)
        end
    end

    inequality_constraints = function (x, θ)
        mapreduce(vcat, inner_inequality_constraints) do constraint
            constraint(x, θ)
        end
    end

    ParametricMPCC(;
        objective,
        equality_constraints,
        inequality_constraints,
        complementarity_constraints = complementarity_constraints[1], # Φ(Gᵢ(x), Hᵢ(x)) ≤ 0, i = 1,...,m
        primal_dimension,
        parameter_dimension = parameter_dimension + 1, # +1 for relaxation parameter
        relaxation_parameter,
        update_parameter,
        max_iterations,
        relaxation_mode,
    )
end
