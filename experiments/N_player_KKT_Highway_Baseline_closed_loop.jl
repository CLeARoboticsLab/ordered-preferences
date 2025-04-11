module N_player_KKT_Highway_Baseline_closed_loop
using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using TrajectoryGamesBase:
    OpenLoopStrategy, unflatten_trajectory, state_dim, control_dim, control_bounds
using GLMakie: GLMakie, Observable
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

function demo(; time_step = 1, verbose = false, num_samples = 10, pareto = false)
    num_players = 3
    dynamics = planar_double_integrator(; control_bounds = (; lb = [-2.0, -2.0], ub = [2.0, 2.0])) # x := (px, py, vx, vy) and u := (ax, ay).
    planning_horizon = 5
    obstacle_radius = 0.25
    collision_avoidance = 0.2

    if !pareto
        penalties = [# 1, α, α²
            #[1.0, 1.0, 1.0],        # α = 1
            #[100.0, 10.0, 1.0],     # α = 10
            #[400.0, 20.0, 1.0],     # α = 20
            #[900.0, 30.0, 1.0],     # α = 30
            #[1600.0, 40.0, 1.0],    # α = 40
            [2500.0, 50.0, 1.0],    # α = 50
            [2500.0, 0.0, 0.0]
        ]
    else
        penalties = [[α^2, α, 1.0] for α = 1:50]
    end

    # Run-time record
    runtime = Float64[]

    # Run the baseline experiment
    @showprogress desc="Running problem instances using baseline..." for ii in [39] 
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
            obstacle_position = Observable([0.25, 0.15])
        
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
                        JLD2.save_object("./data/Highway_closed_loop/Baseline_solution/rfp_$(ii)_B$(penalty_cnt)_T$(time_step)_sol.jld2", solution_dict)
                    end
                end

                (; strategies, solution)
            end
    
            # Load problem data
            problem_data = JLD2.load_object("./data/Highway_closed_loop/problem/rfp_$ii.jld2")
    
            # Player 1
            initial_state1 = Observable(problem_data["initial_state1"]) # (px, py, vx, vy)
            goal_position1 = Observable([0.9, 0.0])
            θ1 = GLMakie.@lift flatten_parameters(; # θ is a flat (column) vector of parameters
                initial_state = $initial_state1,
                goal_position = $goal_position1,
                obstacle_position = $obstacle_position,
            )
    
            # Player 2
            initial_state2 = Observable(problem_data["initial_state2"])
            goal_position2 = Observable([0.9, 0.0])
            θ2 = GLMakie.@lift flatten_parameters(;
                initial_state = $initial_state2,
                goal_position = $goal_position2,
                obstacle_position = $obstacle_position,
            )
    
            # Player 3 
            initial_state3 = Observable(problem_data["initial_state3"])
            goal_position3 = Observable([0.9, 0.0])
            θ3 = GLMakie.@lift flatten_parameters(;
                initial_state = $initial_state3,
                goal_position = $goal_position3,
                obstacle_position = $obstacle_position,
            )
    
            θ = GLMakie.@lift [$θ1..., $θ2..., $θ3...]
    
            println("Solving problem instance #$ii using baseline #$penalty_cnt...")
            println("initial_state1:", initial_state1)
            println("initial_state2:", initial_state2)
            println("initial_state3:", initial_state3)
    
            strategy = GLMakie.@lift let 
                result = get_receding_horizon_solution($θ, ii, penalty_cnt; warmstart_solution)
                warmstart_solution = nothing
                result.strategies
            end
    
            # Plot using GLMakie
            figure = GLMakie.Figure()
            axis = GLMakie.Axis(figure[1, 1];xgridvisible = false, ygridvisible = false, limits = ((-0.5, 1),(-0.25, 0.25)))
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
            baseline_strategy1_xs = GLMakie.@lift([vcat(v[2], v[1], v[3:end]) for v in $strategy[1].xs])
            baseline_strategy2_xs = GLMakie.@lift([vcat(v[2], v[1], v[3:end]) for v in $strategy[2].xs])
            baseline_strategy3_xs = GLMakie.@lift([vcat(v[2], v[1], v[3:end]) for v in $strategy[3].xs])
            # NOTE: The previous code is a workaround to plot the trajectories correctly.
            strategy1 = GLMakie.@lift OpenLoopStrategy($baseline_strategy1_xs, $strategy[1].us)
            strategy2 = GLMakie.@lift OpenLoopStrategy($baseline_strategy2_xs, $strategy[2].us)
            strategy3 = GLMakie.@lift OpenLoopStrategy($baseline_strategy3_xs, $strategy[3].us)
            GLMakie.plot!(axis, strategy1, color = :blue)
            GLMakie.plot!(axis, strategy2, color = :red)
            GLMakie.plot!(axis, strategy3, color = :green)

            # closed_loop + receding horizon demo
            while time_step < 15
                GLMakie.save("data/Highway_closed_loop/Baseline_plots/baseline_$(penalty_cnt)_trajectory$(time_step).png", figure)
            
                # Update the positions of the vehicles
                time_step += 1
                println("Update initial state1")
                θ1.val[1:state_dim(dynamics)] = first(strategy[]).xs[begin + 1] # Asynchronous update: mutate p1's initial state without triggering others
                println("Update initial state2")
                θ2.val[1:state_dim(dynamics)] = strategy[][2].xs[begin + 1]
                println("Update initial state3")
                initial_state3[] = strategy[][3].xs[begin + 1]
            end
            
            # Update penalty_cnt
            penalty_cnt += 1

            # Initialize time_step
            time_step = 1
        end
    end

end

end
