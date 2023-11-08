module OrderedPreferences

using ParametricMCPs
using Symbolics
using BlockArrays 

include("parametric_optimization_problem.jl")
export ParametricOptimizationProblem, solve, total_dim

end # module OrderedPreferences
