module OrderedPreferences

using LinearAlgebra

using ParametricMCPs: ParametricMCPs, ParametricMCP
using Symbolics: Symbolics
using BlockArrays: BlockArrays, BlockArray, Block, blocks
using DelimitedFiles: readdlm

include("parametric_game_penalty_baseline.jl")
export ParametricGamePenalty, solve_penalty, total_dim

end # module OrderedPreferences
