module OrderedPreferences

using LinearAlgebra

using ParametricMCPs: ParametricMCPs, ParametricMCP
using Symbolics: Symbolics
using BlockArrays: BlockArrays, BlockArray, Block

include("parametric_optimization_problem.jl")
export ParametricOptimizationProblem, solve, total_dim

include("parametric_ordered_preferences.jl")
export build_ordered_preferences_problem

include("solve_ordered_preferences.jl")
export solve_ordered_preferences

end # module OrderedPreferences
