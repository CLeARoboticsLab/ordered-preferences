# Blueprint sketch work 

# 1. Define a problem struct that includes objective function, constraints with priorities, 
# lower and upper bounds, initial guess, and problem parameters. 
struct OrderedPreference{T1, T2, T3}
    """
    Defines an optimization with ordered preferrances. 
    Constraints are encoded with the decreasing order of importance.

    Members:
    - objective::T1
    - constraints::T2 <- inequality vs equality separate
    - parameters::T3

    Generate a sequence of parametric MCPs

    """
    # TODO

end

# How does constraints will look like? 
# g₁(x) ≥ 0   <->    initial state + state dynamics (HARD)
# g₂(x) ≥ 0   <->    Don't collide with human
# g₃(x) ≥ 0   <->    Don't collide with the wall
# g₄(x) ≥ 0   <->    Don't break too much


# 2. Define a solve function that takes a problem struct, options like verbose
# and returns a solution struct.
function solve(problem::OrderedPreferrance; initial_guess, verbose, options...)
    """
    Solves an optimization problem with ordered preferrances. 

    Arguments:
    - problem::OrderedPreferrance
    - initial_guess::Vector{Float64}
    - verbose::Bool
    - options...

    Returns:
    - (; z_opt, status, info))
    """
    # TODO

end