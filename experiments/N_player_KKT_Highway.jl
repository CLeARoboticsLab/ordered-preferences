module N_player_KKT_Highway

using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using TrajectoryGamesBase:
    OpenLoopStrategy, unflatten_trajectory, state_dim, control_dim, control_bounds
using CairoMakie: CairoMakie
using BlockArrays
using JLD2, ProgressMeter, Distributions, Random

using OrderedPreferences

function get_setup(num_players; dynamics = UnicycleDynamics, planning_horizon = 20, obstacle_radius = 0.25,  collision_avoidance = 0.2, relaxation_mode = :standard)
    state_dimension = state_dim(dynamics)
    control_dimension = control_dim(dynamics)
    primals_per_agent = (state_dimension + control_dimension) * planning_horizon
    primal_dimensions = fill(primals_per_agent, num_players)
    parameter_dimensions = fill(state_dimension + 4, num_players) # (state, goal, obstacle)

    dummy_primals = BlockArray(zeros(sum(primal_dimensions)), primal_dimensions)
    dummy_parameters = BlockArray(zeros(sum(parameter_dimensions)), parameter_dimensions)

    unflatten_parameters = function (θ)
        θ_iter = Iterators.Stateful(θ)
        initial_state = first(θ_iter, state_dimension)
        goal_position = first(θ_iter, 2)
        obstacle_position = first(θ_iter, 2)
        (; initial_state, goal_position, obstacle_position)
    end

    function flatten_parameters(; initial_state, goal_position, obstacle_position)
        vcat(initial_state, goal_position, obstacle_position)
    end

    objectives = [
        function (z, θ)
            (; xs, us) = unflatten_trajectory(z[Block(i)], state_dimension, control_dimension)
            0.5*sum(sum(u .^ 2) for u in us)
        end
        for i in 1:num_players
    ]

    equality_constraints = [ # private: z[Block(1)] are original private primals
        function (z, θ)
            (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
            (; initial_state) = unflatten_parameters(θ[Block(i)])
            initial_state_constraint = xs[1] - initial_state
            dynamics_constraints = mapreduce(vcat, 2:length(xs)) do k
                xs[k] - dynamics(xs[k - 1], us[k - 1], k)
        end
        vcat(initial_state_constraint, dynamics_constraints)
        end
        for i in 1:num_players
    ] 
    equality_dimensions = [length(equality_constraints[i](dummy_primals, dummy_parameters)) for i in 1:num_players]

    inequality_constraints = [
        function (z, θ)
            (; lb, ub) = control_bounds(dynamics)
            lb_mask = findall(!isinf, lb)
            ub_mask = findall(!isinf, ub)
            (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
            vcat(
                # control bounds (box)
                mapreduce(vcat, us) do u
                    vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
                end,

                # stay within the playing field (for planar_double_integrator)
                mapreduce(vcat, 1:length(xs)) do k
                    px, py, vx, vy = xs[k]
                    position_constraints = vcat(px + 1.0, -px + 1.0, py + 0.2, -py + 0.2)
                    vcat(position_constraints)
                end
            )
        end
        for _ in 1:num_players
    ]
    inequality_dimensions = [length(inequality_constraints[i](dummy_primals, dummy_parameters)) for i in 1:num_players]

    prioritized_preferences = [
        [
            # reach the goal.
            function (z, θ)
                (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
                (; goal_position) = unflatten_parameters(θ[Block(1)]) # Player 1 θ[Block(i)]
                xs[end][1] - goal_position[1] # px[end] ≥ 0.9
            end,

            # Speed limit
            function (z,θ)
                (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
                mapreduce(vcat, 1:length(xs)) do k
                    px, py, vx, vy = xs[k]
                    velocity_constraints = vcat(vx + 0.2, -vx + 0.2, vy + 0.2, -vy + 0.2)
                    vcat(velocity_constraints)
                end
            end,
        ],
        [
            # Speed limit
            function (z,θ)
                (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
                mapreduce(vcat, 1:length(xs)) do k
                    px, py, vx, vy = xs[k]
                    velocity_constraints = vcat(vx + 0.2, -vx + 0.2, vy + 0.2, -vy + 0.2)
                    vcat(velocity_constraints)
                end
            end,

            # reach the goal.
            function (z, θ)
                (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
                (; goal_position) = unflatten_parameters(θ[Block(2)]) # Player 2
                xs[end][1] - goal_position[1] # px[end] ≥ 0.9
            end,
        ],
        [
            # Speed limit
            function (z,θ)
                (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
                mapreduce(vcat, 1:length(xs)) do k
                    px, py, vx, vy = xs[k]
                    velocity_constraints = vcat(vx + 0.2, -vx + 0.2, vy + 0.2, -vy + 0.2)
                    vcat(velocity_constraints)
                end
            end,

            # reach the goal.
            function (z, θ)
                (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
                (; goal_position) = unflatten_parameters(θ[Block(3)]) # Player 3
                xs[end][1] - goal_position[1] # px[end] ≥ 0.9
            end,
        ]
    ]

    # Specify prioritized constraint
    is_prioritized_constraint = [[true, true], [true, true], [true, true]]

    # Shared constraints
    function shared_equality_constraints(z, θ)
        [0]
    end

    function shared_inequality_constraints(z, θ)
        trajectories = map(i -> unflatten_trajectory(z[Block(i)], state_dimension, control_dimension), 1:num_players)
        xs = map(trajectory -> trajectory.xs, trajectories)
        @assert length(xs) == num_players
        # Avoid collision between 3 players
        mapreduce(vcat, 2:length(xs[1])) do k
            [
                sum((xs[1][k][1:2] - xs[2][k][1:2]) .^ 2) - collision_avoidance^2
                sum((xs[1][k][1:2] - xs[3][k][1:2]) .^ 2) - collision_avoidance^2
                sum((xs[2][k][1:2] - xs[3][k][1:2]) .^ 2) - collision_avoidance^2
            ]
        end
    end

    problem = ParametricOrderedPreferencesMPCCGame(;
        objectives,
        equality_constraints,
        inequality_constraints,
        prioritized_preferences,
        is_prioritized_constraint,
        shared_equality_constraints,
        shared_inequality_constraints,
        primal_dimensions,
        parameter_dimensions,
        equality_dimensions,
        inequality_dimensions,
        relaxation_mode,
    )

    (; problem, flatten_parameters, equality_constraints, inequality_constraints, shared_equality_constraints, shared_inequality_constraints, prioritized_preferences)
end

function demo(; verbose = false, num_samples = 10, check_equilibrium = false, filename = "N_player_GOOP_v1.mp4")
    # Algorithm setting
    ϵ = 1.1
    κ = 0.1
    max_iterations = 7
    tolerance = 1e-1
    relaxation_mode = :standard

    num_players = 3
    dynamics = planar_double_integrator(; control_bounds = (; lb = [-2.0, -2.0], ub = [2.0, 2.0])) # x := (px, py, vx, vy) and u := (ax, ay).
    planning_horizon = 5
    obstacle_radius = 0.25
    collision_avoidance = 0.2

    (; problem, flatten_parameters, equality_constraints, inequality_constraints,
        shared_equality_constraints, shared_inequality_constraints, prioritized_preferences) = get_setup(
        num_players;
        dynamics,
        planning_horizon,
        obstacle_radius,
        collision_avoidance,
        relaxation_mode)

    warmstart_solution = nothing

    dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
    primal_dimension = dynamics_dimension * planning_horizon

    # Common obstacle
    obstacle_position = [0.25, 0.15]

    # Tracking not-converged instances
    GOOP_not_converged = []

    # Distribution for sampling feasible trajectories
    Random.seed!(123)
    dist = Normal(0.0, 0.01)
    num_perturb = 20
    equilibrium_tally_goop = []
    tol = 2e-2 

    function get_receding_horizon_solution(θ, ii; warmstart_solution)
        (; relaxation, solution, residual) =
            solve_relaxed_pop_game(problem, warmstart_solution, θ; ϵ, κ, max_iterations, tolerance, verbose)

        if all([string(solution[i].status) != "MCP_Solved" for i in 1:first(size(solution))])
            println("GOOP could not find a solution...moving on to the next problem")
            return nothing
        else
            # Choose the solution with best complementarity residual
            min_residual_idx = argmin(residual)
            println("residual: ", residual[min_residual_idx])
            println("relaxation: ", relaxation[min_residual_idx])
            println("slacks: ", solution[min_residual_idx].slacks)
            strategies = mapreduce(vcat, 1:num_players) do i
                unflatten_trajectory(solution[min_residual_idx].primals[i][1:primal_dimension], state_dim(dynamics), control_dim(dynamics))
            end
            # Main.@infiltrate
            # Save solution
            solution_dict = Dict(
                "residual" => residual[min_residual_idx],
                "relaxation" => relaxation[min_residual_idx],
                "slacks" => solution[min_residual_idx].slacks,
                "strategy1" => strategies[1],
                "strategy2" => strategies[2],
                "strategy3" => strategies[3],
            )
            JLD2.save_object("./data/relaxably_feasible/GOOP_solution/rfp_$ii"*"_sol.jld2", solution_dict)
        end

        (; strategies, solution)
    end

    # Run the experiment
    @showprogress desc="Running problem instances..." for ii in 1:num_samples
        # Load problem data
        problem_data = JLD2.load_object("./data/relaxably_feasible/problem/rfp_$ii.jld2")

        # Player 1
        initial_state1 = problem_data["initial_state1"]
        goal_position1 = [0.9, 0.0]
        θ1 = flatten_parameters(; # θ is a flat (column) vector of parameters
            initial_state = initial_state1,
            goal_position = goal_position1,
            obstacle_position = obstacle_position,
        )

        # Player 2
        initial_state2 = problem_data["initial_state2"]
        goal_position2 = [0.9, 0.0]
        θ2 = flatten_parameters(; 
            initial_state = initial_state2,
            goal_position = goal_position2,
            obstacle_position = obstacle_position,
        )

        # Player 3
        initial_state3 = problem_data["initial_state3"]
        goal_position3 = [0.9, 0.0]
        θ3 = flatten_parameters(; 
            initial_state = initial_state3,
            goal_position = goal_position3,
            obstacle_position = obstacle_position,
        )

        θ = [θ1..., θ2..., θ3...]

        println("Solving problem instance #$ii...")
        println("initial_state1:", initial_state1)
        println("initial_state2:", initial_state2)
        println("initial_state3:", initial_state3)

        result = get_receding_horizon_solution(θ, ii; warmstart_solution)
        warmstart_solution = nothing
        if isnothing(result)
            strategy = nothing
        else
            strategy = result.strategies
        end

        # If not solved, then continue to next problem instance (#5 cannot)
        if isnothing(strategy)
            push!(GOOP_not_converged, ii)
            continue
        else
            if check_equilibrium
                # Check if the solution is an equilibrium
                count_equilibrium_goop = 0
                goop_z = BlockArray(
                    mapreduce(vcat, 1:num_players) do i
                        mapreduce(vcat, 1:planning_horizon) do k
                            vcat(strategy[i].xs[k], strategy[i].us[k])
                        end
                end, fill(primal_dimension, num_players))

                θ_blocked = BlockArray(θ, fill(Int(length(θ)/num_players), num_players))

                @showprogress desc="  Checking equilibrium..." for jj in 1:num_perturb
                    perturbed_x = [[strategy[1].xs[1]], [strategy[2].xs[1]], [strategy[3].xs[1]]]
                    perturbed_u = [[], [], []]

                    for kk in 1:num_players
                        println("   Checking x*...@perturbation #$jj, player#$kk")

                        perturbed_z_block = []
                        check_inequality, check_shared_equality, check_shared_inequality = false, false, false
                        while !(check_inequality && check_shared_equality && check_shared_inequality)
                            # Step 1: Perturb control sequence u by ω = rand(dist, n_size) and generate perturbed trajectory x
                            for i in 1:planning_horizon-1
                                local u = strategy[kk].us[i] + rand(dist, control_dim(dynamics))
                                push!(perturbed_u[kk], u)
                                push!(perturbed_x[kk], dynamics(perturbed_x[kk][i], u))
                            end
                            push!(perturbed_u[kk], [0.0, 0.0])
                            # Step 2: Check if the perturbed trajectory x satisfies shared constraints and inequality constraints
                            # Rejection sampling. Feasible perturbations.Fix others' strategy constant and perturb one player's strategy
                            perturbed_z_block = mapreduce(vcat, 1:planning_horizon) do i
                                vcat(perturbed_x[kk][i], perturbed_u[kk][i])
                            end
                            check_inequality = all(inequality_constraints[kk](perturbed_z_block, θ_blocked) .≥ -tol)

                            perturbed_z = let
                                z_temp = copy(goop_z)
                                z = blocks(z_temp)
                                z[kk] = perturbed_z_block
                                mortar(z)
                            end
                            check_shared_inequality = all(shared_inequality_constraints(perturbed_z, θ_blocked) .≥ -tol)
                            check_shared_equality = all(shared_equality_constraints(perturbed_z, θ_blocked) .== 0.0)

                            # Initialize perturbed trajectory for next iteration
                            perturbed_x = [[strategy[1].xs[1]], [strategy[2].xs[1]], [strategy[3].xs[1]]]
                            perturbed_u = [[], [], []]
                        end

                        # Step 3: Check if f₃(x*, θ) < f₃(x, θ) in the neighborhood of x*
                        f₃_star = sum(max.(0, -prioritized_preferences[kk][1](goop_z[Block(kk)], θ_blocked)))
                        f₃ = sum(max.(0, -prioritized_preferences[kk][1](perturbed_z_block, θ_blocked)))
                        if f₃_star < f₃
                            println("    f₃(x*, θ) < f₃(x, θ) in the neighborhood of x* for player #$kk")
                        elseif isapprox(f₃_star, f₃, atol = tol)
                            println("    f₃(x*, θ) close to f₃(x, θ) for player #$kk")
                            println("    |f₃(x*, θ) - f₃(x, θ)| = $(abs(f₃_star - f₃))")

                            # Step 4: Check if f₂(x*, θ) > f₂(x, θ) in the neighborhood of x*
                            f₂_star = sum(max.(0, -prioritized_preferences[kk][2](goop_z[Block(kk)], θ_blocked)))
                            f₂ = sum(max.(0, -prioritized_preferences[kk][2](perturbed_z_block, θ_blocked)))
                            if f₂_star < f₂
                                println("    f₂(x*, θ) < f₂(x, θ) in the neighborhood of x* for player #$kk")
                            elseif isapprox(f₂_star, f₂, atol = tol)
                                println("    f₂(x*, θ) close to f₂(x, θ) for player #$kk")
                                println("    |f₂(x*, θ) - f₂(x, θ)| = $(abs(f₂_star - f₂))")
                            else
                                println("    f₂(x*, θ) > f₂(x, θ) SOMETHING IS WRONG for player #$kk")
                            end
                        else
                            println("    f₃(x*, θ) > f₃(x, θ) SOMETHING IS WRONG for player #$kk")
                        end

                        # Step 5: Check if x* is an equilibrium
                        if f₃_star < f₃ || (isapprox(f₃_star, f₃, atol = tol) && (f₂_star < f₂ || isapprox(f₂_star, f₂, atol = tol)))
                            println("   x* is a GOOP equilibrium...@perturbation #$jj for player #$kk")
                            count_equilibrium_goop += 1
                        else
                            println("   x* is not a GOOP equilibrium...@perturbation #$jj for player #$kk")
                        end
                    end
                end
                # Check goop equilibrium data
                println("goop soln for prob #$ii is equilibrium in ", count_equilibrium_goop/num_players, " cases (out of $num_perturb)")
                push!(equilibrium_tally_goop, count_equilibrium_goop/num_players)
            end

            # Store speed data for Highway
            horizontal_speed_data = Vector{Vector{Float64}}[]
            vertical_speed_data = Vector{Vector{Float64}}[]
            openloop_distance1 = Vector{Float64}[]
            openloop_distance2 = Vector{Float64}[]
            openloop_distance3 = Vector{Float64}[]

            # Store openloop speed data
            push!(horizontal_speed_data, [vcat(strategy[1].xs...)[3:4:end], vcat(strategy[2].xs...)[3:4:end], vcat(strategy[3].xs...)[3:4:end]])
            push!(vertical_speed_data, [vcat(strategy[1].xs...)[4:4:end], vcat(strategy[2].xs...)[4:4:end], vcat(strategy[3].xs...)[4:4:end]])
        
            # Store openloop distance data
            push!(openloop_distance1, [sqrt(sum((strategy[1].xs[k][1:2] - strategy[2].xs[k][1:2]) .^ 2)) for k in 1:planning_horizon])
            push!(openloop_distance2, [sqrt(sum((strategy[1].xs[k][1:2] - strategy[3].xs[k][1:2]) .^ 2)) for k in 1:planning_horizon])
            push!(openloop_distance3, [sqrt(sum((strategy[2].xs[k][1:2] - strategy[3].xs[k][1:2]) .^ 2)) for k in 1:planning_horizon])

            # Visualize horizontal speed
            T = 1
            fig = CairoMakie.Figure() # limits = (nothing, (nothing, 0.7))
            ax2 = CairoMakie.Axis(fig[1, 1]; xlabel = "time step", ylabel = "speed", title = "Horizontal Speed")
            CairoMakie.scatterlines!(ax2, 0:planning_horizon-1, horizontal_speed_data[T][1], label = "Vehicle 1", color = :blue)
            CairoMakie.scatterlines!(ax2, 0:planning_horizon-1, horizontal_speed_data[T][2], label = "Vehicle 2", color = :red)
            CairoMakie.scatterlines!(ax2, 0:planning_horizon-1, horizontal_speed_data[T][3], label = "Vehicle 3", color = :green)
            CairoMakie.lines!(ax2, 0:planning_horizon-1, [0.2 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)
            fig[2,1:2] = CairoMakie.Legend(fig, ax2, framevisible = false, orientation = :horizontal)

            # Visualize vertical speed
            ax3 = CairoMakie.Axis(fig[1, 2]; xlabel = "time step", ylabel = "speed", title = "Vertical Speed")
            CairoMakie.scatterlines!(ax3, 0:planning_horizon-1, vertical_speed_data[T][1], label = "Vehicle 1", color = :blue)
            CairoMakie.scatterlines!(ax3, 0:planning_horizon-1, vertical_speed_data[T][2], label = "Vehicle 2", color = :red)
            CairoMakie.scatterlines!(ax3, 0:planning_horizon-1, vertical_speed_data[T][3], label = "Vehicle 3", color = :green)
            CairoMakie.lines!(ax3, 0:planning_horizon-1, [0.2 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)

            CairoMakie.save("./data/relaxably_feasible/GOOP_plots/" * "rfp_GOOP_speed_$ii" * ".png", fig)
            fig

            # Visualize distance bw vehicles , limits = (nothing, (collision_avoidance-0.05, 0.4)) 
            fig = CairoMakie.Figure() # limits = (nothing, (nothing, 0.7))
            ax4 = CairoMakie.Axis(fig[1, 1]; xlabel = "time step", ylabel = "distance", title = "Distance bw vehicles")
            CairoMakie.scatterlines!(ax4, 0:planning_horizon-1, openloop_distance1[T], label = "B/w Agent 1 & Agent 2", color = :black, marker = :star5, markersize = 20)
            CairoMakie.scatterlines!(ax4, 0:planning_horizon-1, openloop_distance2[T], label = "B/w Agent 1 & Agent 3", color = :orange, marker = :diamond, markersize = 20)
            CairoMakie.scatterlines!(ax4, 0:planning_horizon-1, openloop_distance3[T], label = "B/w Agent 2 & Agent 3", color = :purple, marker = :circle, markersize = 20)
            CairoMakie.lines!(ax4, 0:planning_horizon-1, [0.2 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)
            fig[2,1] = CairoMakie.Legend(fig, ax4, framevisible = false, orientation = :horizontal)

            CairoMakie.save("./data/relaxably_feasible/GOOP_plots/" * "rfp_GOOP_distance_$ii" * ".png", fig)
            fig
        end
    end

    # Save not-converged instances
    JLD2.save_object("./data/rfp_GOOP_not_converged.jld2", GOOP_not_converged)

    # Save equilibrium tally
    JLD2.save_object("./data/relaxably_feasible/GOOP_solution/rfp_equilibrium.jld2", equilibrium_tally_goop)
    println("equilibrium tally: ", equilibrium_tally_goop)
end

end
