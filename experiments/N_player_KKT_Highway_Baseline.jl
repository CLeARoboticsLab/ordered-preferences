module N_player_KKT_Highway_Baseline
using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using TrajectoryGamesBase:
    OpenLoopStrategy, unflatten_trajectory, state_dim, control_dim, control_bounds
using CairoMakie: CairoMakie
using BlockArrays
using JLD2, ProgressMeter, BenchmarkTools

using OrderedPreferences

function get_setup(num_players, penalty_weights; dynamics = UnicycleDynamics, planning_horizon = 20, collision_avoidance = 0.2)
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
            (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
            sum(sum(u .^ 2) for u in us)
        end
        for _ in 1:num_players
    ]

    equality_constraints = [ # Unlike before, z[Block(i)] are (original) private primals
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
            [sum((xs[1][k][1:2] - xs[2][k][1:2]) .^ 2) - collision_avoidance^2;
            sum((xs[1][k][1:2] - xs[3][k][1:2]) .^ 2) - collision_avoidance^2;
            sum((xs[2][k][1:2] - xs[3][k][1:2]) .^ 2) - collision_avoidance^2]
        end
    end

    shared_equality_dimension = length(shared_equality_constraints(dummy_primals, dummy_parameters))
    shared_inequality_dimension = length(shared_inequality_constraints(dummy_primals, dummy_parameters))

    problem = ParametricGamePenalty(;
        objectives,
        equality_constraints,
        inequality_constraints,
        prioritized_preferences,
        is_prioritized_constraint,
        shared_equality_constraints,
        shared_inequality_constraints,
        primal_dimensions,
        parameter_dimensions,
        shared_equality_dimension,
        shared_inequality_dimension,
        penalty_weights
    )

    (; problem, flatten_parameters, unflatten_parameters)
end

function demo(; verbose = false, num_samples = 10, pareto = false)
    num_players = 3
    dynamics = planar_double_integrator(; control_bounds = (; lb = [-2.0, -2.0], ub = [2.0, 2.0])) # x := (px, py, vx, vy) and u := (ax, ay).
    planning_horizon = 5
    obstacle_radius = 0.25
    collision_avoidance = 0.2

    if !pareto
        penalties = [# 1, α, α²
            [1.0, 1.0, 1.0],        # α = 1
            [100.0, 10.0, 1.0],     # α = 10
            [400.0, 20.0, 1.0],     # α = 20
            [900.0, 30.0, 1.0],     # α = 30
            [1600.0, 40.0, 1.0],    # α = 40
            [2500.0, 50.0, 1.0],    # α = 50
        ]
    else
        penalties = [[α^2, α, 1.0] for α = 1:50]
    end

    # Run-time record
    runtime = Float64[]

    # Run the baseline experiment
    @showprogress desc="Running problem instances using baseline..." for ii in 1:num_samples #[39,97] #1:num_samples
        penalty_cnt = 1
        println("-----------------------------------------------------")
        for penalty in penalties
            println("Running for penalty weights: ", penalty)
            penalty_weights = [penalty for _ in 1:num_players]

            # Define problem 
            (; problem, flatten_parameters) = get_setup(num_players, penalty_weights; dynamics, planning_horizon, collision_avoidance)

            warmstart_solution = nothing
        
            dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
            primal_dimension = dynamics_dimension * planning_horizon
        
            # Common obstacle
            obstacle_position = [0.25, 0.15]
        
            # Tracking not-converged instances
            Baseline_not_converged = []
    
            function get_receding_horizon_solution(θ, ii, penalty_cnt; warmstart_solution)
                # Measure run time
                elapsed_time = @elapsed begin
                    solution = solve_penalty(problem, θ; initial_guess = warmstart_solution, verbose, return_primals = true)
                end
                push!(runtime, elapsed_time)

                if string(solution.status) != "MCP_Solved"
                    println("Baseline #$penalty_cnt could not find a solution...moving on to the next problem")
                    return nothing
                else
                    strategies = mapreduce(vcat, 1:num_players) do i
                        unflatten_trajectory(solution.primals[i][1:primal_dimension], state_dim(dynamics), control_dim(dynamics))
                    end
                    println("slacks: ", solution.slacks)
                    # Save solution
                    solution_dict = Dict(
                        "slacks" => solution.slacks,
                        "strategy1" => strategies[1],
                        "strategy2" => strategies[2],
                        "strategy3" => strategies[3],
                        "primals" => solution.primals,
                    )
                    if !pareto
                        JLD2.save_object("./data/relaxably_feasible/Baseline_solution/$penalty_cnt/rfp_$ii"*"_sol.jld2", solution_dict)
                    else
                        JLD2.save_object("./data_pareto/Baseline_solution/$ii/rfp_$(ii)_baseline$penalty_cnt"*"_sol.jld2", solution_dict)
                    end
                end

                (; strategies, solution)
            end
    
            # Load problem data
            problem_data = JLD2.load_object("./data/relaxably_feasible/problem/rfp_$ii"*".jld2")
    
            # Player 1
            initial_state1 = problem_data["initial_state1"] # (px, py, vx, vy)
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
    
            println("Solving problem instance #$ii using baseline #$penalty_cnt...")
            println("initial_state1:", initial_state1)
            println("initial_state2:", initial_state2)
            println("initial_state3:", initial_state3)
    
            result = get_receding_horizon_solution(θ, ii, penalty_cnt; warmstart_solution)
            warmstart_solution = nothing
            
            if isnothing(result)
                push!(Baseline_not_converged, ii)
                strategy = nothing
            else
                strategy = result.strategies
            end
    
            # If not solved, then continue to next problem instance
            if isnothing(strategy)
                push!(Baseline_not_converged, ii)
                continue
            else
                if !pareto 
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
        
                    CairoMakie.save("./data/relaxably_feasible/Baseline_plots/$penalty_cnt/" * "rfp_baseline_speed_$ii" * ".png", fig)
                    fig
        
                    # Visualize distance bw vehicles , limits = (nothing, (collision_avoidance-0.05, 0.4)) 
                    fig = CairoMakie.Figure() # limits = (nothing, (nothing, 0.7))
                    ax4 = CairoMakie.Axis(fig[1, 1]; xlabel = "time step", ylabel = "distance", title = "Distance bw vehicles")
                    CairoMakie.scatterlines!(ax4, 0:planning_horizon-1, openloop_distance1[T], label = "B/w Agent 1 & Agent 2", color = :black, marker = :star5, markersize = 20)
                    CairoMakie.scatterlines!(ax4, 0:planning_horizon-1, openloop_distance2[T], label = "B/w Agent 1 & Agent 3", color = :orange, marker = :diamond, markersize = 20)
                    CairoMakie.scatterlines!(ax4, 0:planning_horizon-1, openloop_distance3[T], label = "B/w Agent 2 & Agent 3", color = :purple, marker = :circle, markersize = 20)
                    CairoMakie.lines!(ax4, 0:planning_horizon-1, [0.2 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)
                    fig[2,1] = CairoMakie.Legend(fig, ax4, framevisible = false, orientation = :horizontal)
        
                    CairoMakie.save("./data/relaxably_feasible/Baseline_plots/$penalty_cnt/" * "rfp_baseline_distance_$ii" * ".png", fig)
                    fig
                end
            end

            # Save not-converged instances
            if !pareto
                JLD2.save_object("./data/rfp_baseline_not_converged"*"_$penalty_cnt"*".jld2", Baseline_not_converged)
            else
                JLD2.save_object("./data/rfp_$(ii)_baseline_not_converged"*"_$penalty_cnt"*".jld2", Baseline_not_converged)
            end

            # Update penalty_cnt
            penalty_cnt += 1

            # Reset
            Baseline_not_converged = []
        end
        
        # Save runtime
        JLD2.save_object("./data/relaxably_feasible/runtime/rfp_runtime_$(ii)_baseline.jld2", runtime)
        
        # Reset
        runtime = Float64[]
    end

end

end
