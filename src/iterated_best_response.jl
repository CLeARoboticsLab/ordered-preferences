"""
- `best_response_maps` is a vector of functions where each of them is callable as
  `best_response_map(parameters::Vector{Float64}, initial_guess::Vector{Float64})::Vector{Float64}` and returns a flat vector representation of that player's response
- `initial_trajectory_guesses` is a flat vector that represents the initial values of the decision variables
   Set to `nothing` to use the internal internal guess mechanism
"""
function solve_nash!(
    best_response_maps::Vector{<:Function},
    initial_trajectory_guesses::Vector{Union{Nothing, Vector{Vector{Float64}}}};
    convergence_tolerance::Float64 = 1e-3,
    max_iterations = 50,
    verbose = true,
)
    player_converged = falses(length(best_response_maps))
    solutions = Union{NamedTuple, Nothing}[nothing for _ in 1:length(best_response_maps)]
    opponent_positions_ii = nothing

    for kk in 1:max_iterations
        for (ii, val) in enumerate(best_response_maps) 
            # For now, works for two-player case
            #Main.@infiltrate
            (; strategy, solution) = best_response_maps[ii](solutions[ii], opponent_positions_ii) #initial_trajectory_guess_ii = nothing
            # TODO: Add offset to account for time step shift
            #Main.@infiltrate
            solutions[ii] = solution
            response_ii = reduce(hcat, strategy.xs)[1:2, :] |> eachcol |> collect
            opponent_positions_ii = reduce(vcat, response_ii)
            if !isnothing(initial_trajectory_guesses[ii])
                initial_trajectory_guess_ii = reduce(hcat, initial_trajectory_guesses[ii])[1:2, :] |> eachcol |> collect
            else
                initial_trajectory_guess_ii = [zeros(length(response_ii[1])) for _ in 1:length(response_ii)] 
            end            
            innovation = norm(response_ii .- initial_trajectory_guess_ii)
            initial_trajectory_guesses[ii] = strategy.xs
            player_converged[ii] = innovation <= convergence_tolerance
            verbose && println("Player $ii's innovation at iteration $kk: $innovation")
        end
        if all(player_converged)
            verbose && println("Converged after $kk iterations.")
            break
        end
    end

    if !all(player_converged)
        verbose && println("Did not converge after $max_iterations iterations.")
    end

    initial_trajectory_guesses
end