module N_player_KKT_Plot_Trajectory

using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using TrajectoryGamesBase:
    OpenLoopStrategy, unflatten_trajectory, state_dim, control_dim, control_bounds
using GLMakie: GLMakie, Observable
using BlockArrays, JLD2, ProgressMeter, MathTeXEngine

using OrderedPreferences

function get_setup(num_players; dynamics = UnicycleDynamics, planning_horizon = 20, collision_avoidance = 0.2)
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

    # Specify penalty factors [innermost level, second level, outermost level]
    penalty_factors = [
        [100.0, 10.0, 1.0],
        [100.0, 10.0, 1.0],
        [100.0, 10.0, 1.0]
    ]

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
        penalty_factors
    )

    (; problem, flatten_parameters, unflatten_parameters)
end

function demo(; verbose = false, paused = false, filename = "N_player_KKT_baseline.mp4")
    num_players = 3
    dynamics = planar_double_integrator(; control_bounds = (; lb = [-2.0, -2.0], ub = [2.0, 2.0])) # x := (px, py, vx, vy) and u := (ax, ay).
    planning_horizon = 5
    collision_avoidance = 0.2

    # Main.@infiltrate

    (; problem, flatten_parameters) = get_setup(num_players; dynamics, planning_horizon, collision_avoidance)

    warmstart_solution = nothing

    dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
    primal_dimension = dynamics_dimension * planning_horizon

    function get_receding_horizon_solution(θ; warmstart_solution)
        solution = solve_penalty(problem, θ; initial_guess = warmstart_solution, verbose, return_primals = true)
        strategies = mapreduce(vcat, 1:num_players) do i
            unflatten_trajectory(solution.primals[i][1:primal_dimension], state_dim(dynamics), control_dim(dynamics))
        end

        # For later. 
        # sample_solution = open("solution_primals.txt") do file
        #    [parse.(Float64, split(line, ",")) for line in readlines(file)]
        # end

        (; strategies, solution)
    end

    # Common obstacle
    obstacle_position = Observable([0.25, 0.15])

    # Player 1
    initial_state1 = Observable([-0.2, 0.0, 0.8, 0.0]) # (px, py, vx, vy)
    goal_position1 = Observable([0.9, 0.0])
    θ1 = GLMakie.@lift flatten_parameters(; # θ is a flat (column) vector of parameters
        initial_state = $initial_state1,
        goal_position = $goal_position1,
        obstacle_position = $obstacle_position,
    )

    # Player 2
    initial_state2 = Observable([0.1, 0.05, 0.2, 0.0])
    goal_position2 = Observable([0.9, 0.0])
    θ2 = GLMakie.@lift flatten_parameters(;
        initial_state = $initial_state2,
        goal_position = $goal_position2,
        obstacle_position = $obstacle_position,
    )

    # Player 3
    initial_state3 = Observable([0.4, 0.0, 0.2, 0.0])
    goal_position3 = Observable([0.9, 0.0])
    θ3 = GLMakie.@lift flatten_parameters(;
        initial_state = $initial_state3,
        goal_position = $goal_position3,
        obstacle_position = $obstacle_position,
    )

    θ = GLMakie.@lift [$θ1..., $θ2..., $θ3...]

    println("Player 1's goal_position:", goal_position1)
    println("Player 2's goal_position:", goal_position2)
    println("Player 3's goal_position:", goal_position3)

    strategy = GLMakie.@lift let
        result = get_receding_horizon_solution($θ; warmstart_solution)
        warmstart_solution = nothing

        result.strategies
    end

    # Main.@infiltrate

    figure = GLMakie.Figure()
    axis = GLMakie.Axis(figure[1, 1]; aspect = GLMakie.DataAspect(), limits = ((-1, 1), (-0.2, 0.2)))

    # pause and stop buttons
    figure[2, 1] = buttongrid = GLMakie.GridLayout(tellwidth = false)
    is_paused = GLMakie.Observable(paused)
    buttongrid[1, 1] = pause_button = GLMakie.Button(figure, label = "Pause")
    GLMakie.on(pause_button.clicks) do _
        is_paused[] = !is_paused[]
    end

    is_stopped = GLMakie.Observable(false)
    buttongrid[1, 2] = stop_button = GLMakie.Button(figure, label = "Stop")
    GLMakie.on(stop_button.clicks) do _
        is_stopped[] = !is_stopped[]
    end

    # visualize initial states (account for asynchronous update)
    GLMakie.scatter!(
        axis,
        GLMakie.@lift([GLMakie.Point2f($θ1[1:2]), GLMakie.Point2f($θ2[1:2]), GLMakie.Point2f($θ3[1:2])]),
        markersize = 20,
        color = [:blue, :red, :green],
    )

    # Visualize highway lanes
    GLMakie.lines!(axis, [(-1, 0.2), (1, 0.2)], color = :black)
    GLMakie.lines!(axis, [(-1, -0.2), (1, -0.2)], color = :black)
    GLMakie.lines!(axis, [(-1, 0.0), (1, 0.0)], color = :black, linestyle = :dash)

    # visualize goal positions (common goal for now)
    GLMakie.scatter!(
        axis,
        GLMakie.@lift(GLMakie.Point2f($goal_position1)),
        markersize = 20,
        color = :cyan,
    )

    # Visualize trajectories
    strategy1 = GLMakie.@lift OpenLoopStrategy($strategy[1].xs, $strategy[1].us)
    strategy2 = GLMakie.@lift OpenLoopStrategy($strategy[2].xs, $strategy[2].us)
    strategy3 = GLMakie.@lift OpenLoopStrategy($strategy[3].xs, $strategy[3].us)

    GLMakie.plot!(axis, strategy1, color = :blue)
    GLMakie.plot!(axis, strategy2, color = :red)
    GLMakie.plot!(axis, strategy3, color = :green)

    Main.@infiltrate

end

function plot_trajectories(;num_samples = 100, num_penalty = 6)
    baseline_label = ["1", "10", "20", "30", "40", "50"]
    baseline_data = Dict[]

    @showprogress for ii in 1:num_samples #setdiff(1:num_samples, [23, 24, 25, 26, 27, 28])
        # 1. Plot GOOP
        problem_data = JLD2.load_object("./data/relaxably_feasible/problem/rfp_$ii.jld2")
        filename = "rfp_$(ii)_sol.jld2"
        # goop_data = load_object("data/relaxably_feasible/GOOP_solution/$filename")

        goal_position1 = Observable([0.9, 0.0])
        initial_state1 = Observable(problem_data["initial_state1"])
        initial_state2 = Observable(problem_data["initial_state2"])
        initial_state3 = Observable(problem_data["initial_state3"])

        # goop_strategy1 = Observable(goop_data["strategy1"])
        # goop_strategy2 = Observable(goop_data["strategy2"])
        # goop_strategy3 = Observable(goop_data["strategy3"])

        figure = GLMakie.Figure(size = (400, 500)) #(5,4)
        axis = GLMakie.Axis(figure[1, 1]; aspect = GLMakie.DataAspect(), limits = ((-0.5, 1), (-0.25, 0.25)), yticks = [-0.2, 0.2], xticks = [0.0, 1.0])
        axis.xgridvisible = false
        axis.ygridvisible = false
        GLMakie.hidespines!(axis)
        GLMakie.hidexdecorations!(axis)
        GLMakie.hideydecorations!(axis)

        # add text
        GLMakie.text!(axis, 0.0, 0.35, text="Ours (GOOP)", 
            font=:bold, align=(:left, :top), offset=(4, -2), 
            space=:relative, fontsize = 18)

        GLMakie.text!(axis, 0.84, 0.8, text="Goal",
            font=:bold, align=(:left, :top), offset=(4, -2), 
            space=:relative, fontsize = 18, color=:black)

        # visualize initial states (account for asynchronous update)
        GLMakie.scatter!(
            axis,
            GLMakie.@lift([GLMakie.Point2f($initial_state1), GLMakie.Point2f($initial_state2), GLMakie.Point2f($initial_state3)]),
            markersize = 20,
            color = [:blue, :red, :green],
            label=["Vehicle 1 (ambulance)", "Vehicle 2 (passenger car)", "Vehicle 3 (passenger car)"]
        )

        # Visualize highway lanes
        GLMakie.lines!(axis, [(-1, 0.2), (1, 0.2)], color = :black)
        GLMakie.lines!(axis, [(-1, -0.2), (1, -0.2)], color = :black)
        GLMakie.lines!(axis, [(-1, 0.0), (1, 0.0)], color = :black, linestyle = :dash)

        # visualize goal positions (common goal for now)
        GLMakie.scatter!(
            axis,
            GLMakie.@lift(GLMakie.Point2f($goal_position1)),
            markersize = 20,
            marker = :star5,
            color = :grey,
        )

        # Visualize trajectories
        warmstart_samples = 10
        for jj in 1:warmstart_samples 
            try
                filename_w = "rfp_$(ii)_w$(jj)_sol.jld2"
                goop_data = load_object("data/relaxably_feasible/GOOP_solution/$filename_w")
                goop_strategy1 = Observable(goop_data["strategy1"])
                goop_strategy2 = Observable(goop_data["strategy2"])
                goop_strategy3 = Observable(goop_data["strategy3"])
                goop_strategy1 = GLMakie.@lift OpenLoopStrategy($goop_strategy1.xs, $goop_strategy1.us)
                goop_strategy2 = GLMakie.@lift OpenLoopStrategy($goop_strategy2.xs, $goop_strategy2.us)
                goop_strategy3 = GLMakie.@lift OpenLoopStrategy($goop_strategy3.xs, $goop_strategy3.us)
                
                GLMakie.plot!(axis, goop_strategy1, color = :blue)
                GLMakie.plot!(axis, goop_strategy2, color = :red)
                GLMakie.plot!(axis, goop_strategy3, color = :green)
            catch e
                println("Error: $e")
            end
        end

        # 2. Plot Baseline (only 2,4,6)
        for jj in 1:num_penalty
            push!(baseline_data, load_object("data/relaxably_feasible/Baseline_solution/$jj/$filename"))
        end
        for jj in setdiff(1:num_penalty, [1,3,5])

            axis = GLMakie.Axis(figure[Int(1+jj/2), 1]; aspect = GLMakie.DataAspect(), limits = ((-0.5, 1), (-0.25, 0.25)), yticks = [-0.2, 0.2], xticks = [0.0, 1.0])
            axis.xgridvisible = false
            axis.ygridvisible = false
            GLMakie.hidespines!(axis)
            GLMakie.hidexdecorations!(axis)
            GLMakie.hideydecorations!(axis)

            baseline_strategy1 = Observable(baseline_data[jj]["strategy1"])
            baseline_strategy2 = Observable(baseline_data[jj]["strategy2"])
            baseline_strategy3 = Observable(baseline_data[jj]["strategy3"])

            # add text
            baseline = baseline_label[jj] 
            GLMakie.text!(axis, 0.0, 0.35, text="Baseline (α = $baseline)",
            font=:bold, align=(:left, :top), offset=(4, -2), 
            space=:relative, fontsize = 18)

            # visualize initial states (account for asynchronous update)
            GLMakie.scatter!(
                axis,
                GLMakie.@lift([GLMakie.Point2f($initial_state1), GLMakie.Point2f($initial_state2), GLMakie.Point2f($initial_state3)]),
                markersize = 20,
                color = [:blue, :red, :green],
            )

            # Visualize highway lanes
            GLMakie.lines!(axis, [(-1, 0.2), (1, 0.2)], color = :black)
            GLMakie.lines!(axis, [(-1, -0.2), (1, -0.2)], color = :black)
            GLMakie.lines!(axis, [(-1, 0.0), (1, 0.0)], color = :black, linestyle = :dash)

            # visualize goal positions (common goal for now)
            GLMakie.scatter!(
                axis,
                GLMakie.@lift(GLMakie.Point2f($goal_position1)),
                markersize = 20,
                marker = :star5,
                color = :grey,
            )

            # Visualize trajectories
            baseline_strategy1 = GLMakie.@lift OpenLoopStrategy($baseline_strategy1.xs, $baseline_strategy1.us)
            baseline_strategy2 = GLMakie.@lift OpenLoopStrategy($baseline_strategy2.xs, $baseline_strategy2.us)
            baseline_strategy3 = GLMakie.@lift OpenLoopStrategy($baseline_strategy3.xs, $baseline_strategy3.us)

            GLMakie.plot!(axis, baseline_strategy1, color = :blue)
            GLMakie.plot!(axis, baseline_strategy2, color = :red)
            GLMakie.plot!(axis, baseline_strategy3, color = :green)
        end

        # Save plot
        GLMakie.save("data/relaxably_feasible/GOOP_Baseline_trajectory/$filename"[1:end-4] * "_trajectory" * ".png", figure)
        
        # Initialize baseline_data for next round
        baseline_data = Dict[]
    end

end

end