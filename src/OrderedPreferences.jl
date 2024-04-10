module OrderedPreferences

using LinearAlgebra

using ParametricMCPs: ParametricMCPs, ParametricMCP
using Symbolics: Symbolics
using BlockArrays: BlockArrays, BlockArray, Block

include("parametric_optimization_problem.jl")
export ParametricOptimizationProblem, solve, total_dim

include("parametric_ordered_preferences.jl")
export ParametricOrderedPreferencesProblem, solve

include("parametric_MPCC.jl")
export ParametricMPCC, solve_relaxed_mpcc

include("parametric_ordered_preferences_MPCC.jl")
export ParametricOrderedPreferencesMPCC

end # module OrderedPreferences
