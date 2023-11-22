using Dictionaries: Dictionaries, Dictionary, dictionary, sortkeys
using OrderedPreferences

"""
Infeasible LP
min₍ₓ₁,ₓ₂₎  x₁ + x₂
s.t.        g₁(x₁, x₂) = x₁ ≥ 6,
            g₂(x₁, x₂) = x₂ ≥ 6
            g₃(x₁, x₂) = x₁ + x₂ ≤ 11
Assume g₃ more important than g₂, i.e. innermost problem minimizes slack for g₃.
"""
function simple_linear()

    # Define Original (infeasible) Problem
    objective(x, θ) = sum(x)
    equality_constraints(x, θ) = []
    inequality_constraints(x, θ) = [x[1] - 6.0]
    prioritized_inequality_constraints = dictionary([ #
        1 => function (x, θ)
            [x[2] - 6.0]
        end,
        2 => function (x, θ)
            [-x[1] - x[2] + 11.0]
        end,
    ])

    ParametricOrderedPreferencesProblem(;
        objective,
        equality_constraints,
        inequality_constraints,
        prioritized_inequality_constraints,
        primal_dimension = 2,
        parameter_dimension = 0,
    )
end
