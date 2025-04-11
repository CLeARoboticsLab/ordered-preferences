using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using TrajectoryGamesBase:
    OpenLoopStrategy, unflatten_trajectory, state_dim, control_dim, control_bounds
using GLMakie: GLMakie, Observable
using JLD2, ProgressMeter

using OrderedPreferences

include("N_player_KKT_Highway.jl")


function demo(; time_step = 1, verbose = false)
    # Algorithm setting
    ϵ = 1.1
    κ = 0.5
    max_iterations = 9
    tolerance = 5e-2
    relaxation_mode = :standard

    num_players = 3
    control_bounds = (; lb = [-2.0, -2.0], ub = [2.0, 2.0])
    dynamics = planar_double_integrator(; control_bounds) # x := (px, py, vx, vy) and u := (ax, ay).
    planning_horizon = 5
    obstacle_radius = 0.25
    collision_avoidance = 0.2

    (; problem, flatten_parameters) = N_player_KKT_Highway.get_setup(
        num_players;
        dynamics,
        planning_horizon,
        obstacle_radius,
        collision_avoidance,
        relaxation_mode)

    warmstart_solution = nothing

    dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
    primal_dimension = dynamics_dimension * planning_horizon


    function get_receding_horizon_solution(θ, ii, jj; warmstart_solution)
        println("Solving for time step $(time_step)...")
        # Measure run time
        elapsed_time = @elapsed begin
            (; relaxation, solution, residual) =
                solve_relaxed_pop_game(problem, warmstart_solution, θ; ϵ, κ, max_iterations, tolerance, verbose)
        end
        println("Elapsed time: ", elapsed_time)

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
            # Save solution
            solution_dict = Dict(
                "residual" => residual[min_residual_idx],
                "relaxation" => relaxation[min_residual_idx],
                "slacks" => solution[min_residual_idx].slacks,
                "strategy1" => strategies[1],
                "strategy2" => strategies[2],
                "strategy3" => strategies[3],
                "primals" => solution[min_residual_idx].primals,
            )
            JLD2.save_object("./data/Highway_closed_loop/GOOP_solution/Highway_$(ii)_w$(jj)_T$(time_step)"*"_sol.jld2", solution_dict)
        end

        (; strategies, solution)
    end

    # Run the experiment
    @showprogress desc="Running problem instances..." for ii in [39]
        # Load problem data
        problem_data = JLD2.load_object("./data/Highway_closed_loop/problem/rfp_$ii.jld2")

        obstacle_position = Observable([0.25, 0.15]) # placeholder
        # Player 1
        initial_state1 = Observable(problem_data["initial_state1"])
        goal_position1 = Observable([0.9, 0.0])
        θ1 = GLMakie.@lift flatten_parameters(; # θ is a flat (column) vector of parameters
            initial_state = $initial_state1,
            goal_position = $goal_position1,
            obstacle_position = $obstacle_position,
        )

        # Player 2
        initial_state2 = Observable(problem_data["initial_state2"])
        goal_position2 = Observable([0.9, 0.1])
        θ2 = GLMakie.@lift flatten_parameters(; 
            initial_state = $initial_state2,
            goal_position = $goal_position2,
            obstacle_position = $obstacle_position,
        )

        # Player 3
        initial_state3 = Observable(problem_data["initial_state3"])
        goal_position3 = Observable([0.9, -0.1])
        θ3 = GLMakie.@lift flatten_parameters(; 
            initial_state = $initial_state3,
            goal_position = $goal_position3,
            obstacle_position = $obstacle_position,
        )

        θ = GLMakie.@lift [$θ1..., $θ2..., $θ3...]

        println("Solving problem instance #$ii...")
        println("initial_state1:", initial_state1)
        println("initial_state2:", initial_state2)
        println("initial_state3:", initial_state3)

        # Generate multiple equilibrium solutions
        warmstart_samples = 1
        @showprogress desc="    Using different initial guesses..." for jj in 1:warmstart_samples
            if jj == 1
                # initial guess is all zeros
                warmstart_solution = nothing
            else
                # using a random control sequence
                warmstart_x = [[initial_state1], [initial_state2], [initial_state3]]
                warmstart_u = [[], [], []]
                warmstart_solution = []
                # rand_u = 4 * rand(num_players, control_dim(dynamics)) .- 2.0 # between -2.0 and 2.0
                rand_u = hcat(2.0*rand(num_players), 4*rand(num_players) .- 2.0) # ax > 0, -2.0 < ay < 2.0
                for kk in 1:num_players
                    for i in 1:planning_horizon-1
                        push!(warmstart_u[kk], rand_u[kk,:])
                        push!(warmstart_x[kk], dynamics(warmstart_x[kk][i], rand_u[kk,:]))
                    end
                    push!(warmstart_u[kk], [0.0, 0.0])
                    warmstart_primals = mapreduce(vcat, 1:planning_horizon) do i 
                        vcat(warmstart_x[kk][i], warmstart_u[kk][i])
                    end
                    push!(warmstart_solution, warmstart_primals)
                end
                warmstart_solution = vcat(warmstart_solution...)
            end

            strategy = GLMakie.@lift let
                result = get_receding_horizon_solution($θ, ii, jj; warmstart_solution)
                warmstart_solution = nothing
                result.strategies
            end

            # Plot using GLMakie
            figure = GLMakie.Figure()
            axis = GLMakie.Axis(figure[1, 1]; aspect = DataAspect(), xgridvisible = false, ygridvisible = false, limits = ((-0.5, 1),(-0.25, 0.25)))
            GLMakie.hidedecorations!(axis)
            GLMakie.hidespines!(axis)

            # visualize initial states (account for asynchronous update)
            GLMakie.scatter!(
                axis,
                GLMakie.@lift([GLMakie.Point2f($θ1[1:2]), GLMakie.Point2f($θ2[1:2]), GLMakie.Point2f($θ3[1:2])]),
                markersize = 20,
                color = [:blue, :red, :green],
                label=["Vehicle 1 (ambulance)", "Vehicle 2 (passenger car)", "Vehicle 3 (passenger car)"]
            )

            # Visualize highway lanes
            GLMakie.lines!(axis, [(-1, 0.25), (1, 0.25)], color = :black)
            GLMakie.lines!(axis, [(-1, 0.0), (1, 0.0)], color = :black, linestyle = :dash)
            GLMakie.lines!(axis, [(-1, -0.25), (1, -0.25)], color = :black)

            # visualize goal positions (common goal for now)
            GLMakie.scatter!(
                axis,
                GLMakie.@lift(GLMakie.Point2f($goal_position1)),
                markersize = 20,
                marker = :star5,
                color = :grey,
            )

            # Visualize trajectories
            # NOTE: For some reason, GLMakie plots trajectories after reflecting the points along y=x. 
            # To fix this, we need to reflect the points along y=x before plotting them.
            goop_strategy1_xs = GLMakie.@lift([vcat(v[2], v[1], v[3:end]) for v in $strategy[1].xs])
            goop_strategy2_xs = GLMakie.@lift([vcat(v[2], v[1], v[3:end]) for v in $strategy[2].xs])
            goop_strategy3_xs = GLMakie.@lift([vcat(v[2], v[1], v[3:end]) for v in $strategy[3].xs])
            # NOTE: The previous code is a workaround to plot the trajectories correctly.
            strategy1 = GLMakie.@lift OpenLoopStrategy($goop_strategy1_xs, $strategy[1].us)
            strategy2 = GLMakie.@lift OpenLoopStrategy($goop_strategy2_xs, $strategy[2].us)
            strategy3 = GLMakie.@lift OpenLoopStrategy($goop_strategy3_xs, $strategy[3].us)
            GLMakie.plot!(axis, strategy1, color = :blue)
            GLMakie.plot!(axis, strategy2, color = :red)
            GLMakie.plot!(axis, strategy3, color = :green)

            # closed_loop + receding horizon demo
            while time_step < 15
                GLMakie.save("data/Highway_closed_loop/GOOP_plots/trajectory$(time_step).png", figure)
            
                # Update the positions of the vehicles
                time_step += 1
                println("Update initial state1")
                θ1.val[1:state_dim(dynamics)] = first(strategy[]).xs[begin + 1] # Asynchronous update: mutate p1's initial state without triggering others
                println("Update initial state2")
                θ2.val[1:state_dim(dynamics)] = strategy[][2].xs[begin + 1]
                println("Update initial state3")
                initial_state3[] = strategy[][3].xs[begin + 1]
            end
        end
    end
end



# # Store speed data for Highway
# horizontal_speed_data = Vector{Vector{Float64}}[]
# vertical_speed_data = Vector{Vector{Float64}}[]
# openloop_distance1 = Vector{Float64}[]
# openloop_distance2 = Vector{Float64}[]
# openloop_distance3 = Vector{Float64}[]

# # Store openloop speed data
# push!(horizontal_speed_data, [vcat(strategy[1].xs...)[3:4:end], vcat(strategy[2].xs...)[3:4:end], vcat(strategy[3].xs...)[3:4:end]])
# push!(vertical_speed_data, [vcat(strategy[1].xs...)[4:4:end], vcat(strategy[2].xs...)[4:4:end], vcat(strategy[3].xs...)[4:4:end]])

# # Store openloop distance data
# push!(openloop_distance1, [sqrt(sum((strategy[1].xs[k][1:2] - strategy[2].xs[k][1:2]) .^ 2)) for k in 1:planning_horizon])
# push!(openloop_distance2, [sqrt(sum((strategy[1].xs[k][1:2] - strategy[3].xs[k][1:2]) .^ 2)) for k in 1:planning_horizon])
# push!(openloop_distance3, [sqrt(sum((strategy[2].xs[k][1:2] - strategy[3].xs[k][1:2]) .^ 2)) for k in 1:planning_horizon])

# # Visualize horizontal speed
# T = 1
# fig = CairoMakie.Figure() # limits = (nothing, (nothing, 0.7))
# ax2 = CairoMakie.Axis(fig[1, 1]; xlabel = "time step", ylabel = "speed", title = "Horizontal Speed")
# CairoMakie.scatterlines!(ax2, 0:planning_horizon-1, horizontal_speed_data[T][1], label = "Vehicle 1", color = :blue)
# CairoMakie.scatterlines!(ax2, 0:planning_horizon-1, horizontal_speed_data[T][2], label = "Vehicle 2", color = :red)
# CairoMakie.scatterlines!(ax2, 0:planning_horizon-1, horizontal_speed_data[T][3], label = "Vehicle 3", color = :green)
# CairoMakie.lines!(ax2, 0:planning_horizon-1, [0.2 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)
# fig[2,1:2] = CairoMakie.Legend(fig, ax2, framevisible = false, orientation = :horizontal)

# # Visualize vertical speed
# ax3 = CairoMakie.Axis(fig[1, 2]; xlabel = "time step", ylabel = "speed", title = "Vertical Speed")
# CairoMakie.scatterlines!(ax3, 0:planning_horizon-1, vertical_speed_data[T][1], label = "Vehicle 1", color = :blue)
# CairoMakie.scatterlines!(ax3, 0:planning_horizon-1, vertical_speed_data[T][2], label = "Vehicle 2", color = :red)
# CairoMakie.scatterlines!(ax3, 0:planning_horizon-1, vertical_speed_data[T][3], label = "Vehicle 3", color = :green)
# CairoMakie.lines!(ax3, 0:planning_horizon-1, [0.2 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)

# CairoMakie.save("./data/relaxably_feasible/GOOP_plots/" * "rfp_GOOP_speed_$(ii)_w$jj" * ".png", fig)
# fig

# # Visualize distance bw vehicles , limits = (nothing, (collision_avoidance-0.05, 0.4)) 
# fig = CairoMakie.Figure() # limits = (nothing, (nothing, 0.7))
# ax4 = CairoMakie.Axis(fig[1, 1]; xlabel = "time step", ylabel = "distance", title = "Distance bw vehicles")
# CairoMakie.scatterlines!(ax4, 0:planning_horizon-1, openloop_distance1[T], label = "B/w Agent 1 & Agent 2", color = :black, marker = :star5, markersize = 20)
# CairoMakie.scatterlines!(ax4, 0:planning_horizon-1, openloop_distance2[T], label = "B/w Agent 1 & Agent 3", color = :orange, marker = :diamond, markersize = 20)
# CairoMakie.scatterlines!(ax4, 0:planning_horizon-1, openloop_distance3[T], label = "B/w Agent 2 & Agent 3", color = :purple, marker = :circle, markersize = 20)
# CairoMakie.lines!(ax4, 0:planning_horizon-1, [0.2 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)
# fig[2,1] = CairoMakie.Legend(fig, ax4, framevisible = false, orientation = :horizontal)

# CairoMakie.save("./data/relaxably_feasible/GOOP_plots/" * "rfp_GOOP_distance_$(ii)_w$jj" * ".png", fig)
# fig