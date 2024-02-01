"""
- `best_response_maps` is a vector of functions where each of them is callable as
  `best_response_map(parameters::Vector{Float64}, initial_guess::Vector{Float64})::Vector{Float64}` and returns a flat vector representation of that player's response
- `initial_trajectory_guesses` is a flat vector that represents the initial values of the decision variables
   Set to `nothing` to use the internal internal guess mechanism
"""
function solve_nash!(
    best_response_maps::Vector{<:Function},
    initial_trajectory_guesses::Vector{Union{Nothing, Vector{Vector{Float64}}}},
    convergence_tolerance::Float64 = 1e-3,
    max_iterations = 50,
    verbose = true,
)
    player_converged = falses(length(best_response_maps))

    for kk in 1:max_iterations
        for (ii, val) in enumerate(best_response_maps)
            if !isnothing(initial_trajectory_guesses[ii])
                initial_trajectory_guess_ii = initial_trajectory_guesses[ii]
            else
                initial_trajectory_guess_ii = nothing
            end
            # error in solve() in parametric_ordered_preferences.jl. For now, use built-in guess mechanism
            initial_trajectory_guess_ii = nothing
            (; xs) = best_response_maps[ii](initial_trajectory_guess_ii)
            response_ii = reduce(hcat, xs)[1:2, :] |> eachcol |> collect

            if isnothing(initial_trajectory_guess_ii) 
                initial_trajectory_guess_ii = [zeros(length(response_ii[1])) for _ in 1:length(response_ii)] 
            else 
                initial_trajectory_guess_ii
            end
            innovation = norm(response_ii .- initial_trajectory_guess_ii)
            initial_trajectory_guesses[ii] = response_ii 
            player_converged[ii] = innovation <= convergence_tolerance
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