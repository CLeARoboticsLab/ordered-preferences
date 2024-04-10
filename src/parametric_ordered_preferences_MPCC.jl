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
    preferences_dimension,
    relaxation_parameter,
    update_parameter,
    max_iterations,
    relaxation_mode = :standard, # This is for standard relaxation. #TODO l_infinity mode
)
    # Problem data
    ordered_priority_levels = eachindex(prioritized_preferences)

    fixed_dual_dimension = 0
    inner_equality_constraints = Any[equality_constraints]
    inner_inequality_constraints = Any[inequality_constraints]

    subproblems = ParametricOptimizationProblem[]

    function set_up_level(priority_level)
        #TODO: Implement priority objective (for now) vs priority constraints (slacks are required)
        primal_dimension_ii = primal_dimension + fixed_dual_dimension

        priority_preferences_ii = prioritized_preferences[priority_level]

        # Define symbolic variables for primals.
        # Define symbolic variables for parameters.
        # Build symbolic expression for objective and constraints. 
        # Build complementarity constraints from inequality constraints.
        # Concatenate complementarity constraints into inequality constraints for the MPCC. #TODO: Could be equality constraints)
        # Use θ = [ϵ₀, ϵ₁,... ϵₖ] to address relaxation parameters #TODO: Might contain slacks after ϵₖ  
        # Update dual dimension for the next level.

    end

    for priority_level in ordered_priority_levels
        set_up_level(priority_level)
    end
    set_up_level(nothing)

    ParametricMPCC(;
        objective,
        equality_constraints,
        inequality_constraints,
        complementarity_constraints, # Φ(Gᵢ(x), Hᵢ(x)) ≤ 0, i = 1,...,m
        primal_dimension,   
        parameter_dimension,
        relaxation_parameter,
        update_parameter,
        max_iterations,
        relaxation_mode,
    )
end

