module OrderedPreferences

using LinearAlgebra

using ParametricMCPs: ParametricMCPs, ParametricMCP
using Symbolics: Symbolics
using BlockArrays: BlockArrays, BlockArray, Block, blocks
using DelimitedFiles: readdlm

using JLD2

include("parametric_ordered_preferences_MPCC_game.jl")
export ParametricOrderedPreferencesMPCCGame, solve_relaxed_pop_game, total_dim

include("GOOP_classifier.jl")
export ParametricGameClassifier, classify_game, total_dim

end # module OrderedPreferences
