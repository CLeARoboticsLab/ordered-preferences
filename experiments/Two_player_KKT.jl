module Two_player_KKT

using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using TrajectoryGamesBase:
    OpenLoopStrategy, unflatten_trajectory, state_dim, control_dim, control_bounds
using GLMakie: GLMakie, Observable
using BlockArrays

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
            sum(sum(u .^ 2) for u in us)
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

                # limit acceleration and don't go too fast, stay within the playing field (for planar_double_integrator)
                mapreduce(vcat, 1:length(xs)) do k
                    px, py, vx, vy = xs[k]
                    position_constraints = vcat(px + 1.0, -px + 1.0, py + 1.0, -py + 1.0)
                    vcat(position_constraints)
                end
            )
        end
        for _ in 1:num_players
    ]
    inequality_dimensions = [length(inequality_constraints[i](dummy_primals, dummy_parameters)) for i in 1:num_players]

    prioritized_preferences = [    
        [
            # most important: obstacle avoidance
            function (z, θ)
                (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
                (; obstacle_position) = unflatten_parameters(θ[Block(i)])
                mapreduce(vcat, 2:length(xs)) do k
                    sum((xs[k][1:2] - obstacle_position) .^ 2) - obstacle_radius^2
                end
            end,

            # reach the goal.
            function (z, θ)
                (; xs, us) = unflatten_trajectory(z[Block(1)], state_dimension, control_dimension)
                (; goal_position) = unflatten_parameters(θ[Block(i)])
                goal_deviation = xs[end][1:2] .- goal_position
                [
                    goal_deviation .+ 0.01
                    -goal_deviation .+ 0.01
                ]
            end
        ]
        for i in 1:num_players
    ]

    # Specify priortized constraint
    is_prioritized_constraint = [[true, true], [true, true]]

    # Shared constraints
    function shared_equality_constraints(z, θ)
        [0]
    end

    function shared_inequality_constraints(z, θ)
        trajectories = map(i -> unflatten_trajectory(z[Block(i)], state_dimension, control_dimension), 1:num_players)
        xs = map(trajectory -> trajectory.xs, trajectories)
        @assert length(xs) == num_players
        # Avoid collision between two players
        mapreduce(vcat, 2:length(xs[1])) do k
            sum((xs[1][k][1:2] - xs[2][k][1:2]) .^ 2) - collision_avoidance^2
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

    (; problem, flatten_parameters, unflatten_parameters)
end

function demo(; verbose = false, paused = false, record = false, filename = "Two_player_KKT.mp4")
    # Algorithm setting
    ϵ = 1.1
    κ = 0.1
    max_iterations = 16
    tolerance = 5e-3
    relaxation_mode = :standard

    # dynamics = UnicycleDynamics(; control_bounds = (; lb = 10*[-1.0, -1.0], ub = 10*[1.0, 1.0])) # x := (px, py, v, θ) and u := (a, ω). Need to give initial velocity
    num_players = 2
    dynamics = planar_double_integrator(; control_bounds = (; lb = [-1.0, -1.0], ub = [1.0, 1.0])) # x := (px, py, vx, vy) and u := (ax, ay).
    planning_horizon = 7
    obstacle_radius = 0.25
    collision_avoidance = 0.2

    (; problem, flatten_parameters) = get_setup(num_players; dynamics, planning_horizon, obstacle_radius, collision_avoidance, relaxation_mode)

    warmstart_solution = nothing

    dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
    primal_dimension = dynamics_dimension * planning_horizon

    function get_receding_horizon_solution(θ; warmstart_solution)
        (; relaxation, solution, residual) =
            solve_relaxed_pop_game(problem, warmstart_solution, θ; ϵ, κ, max_iterations, tolerance, verbose)
        println("residual: ", residual)
        println("relaxation: ", relaxation)
        strategies = mapreduce(vcat, 1:num_players) do i
            unflatten_trajectory(solution[end].primals[i][1:primal_dimension], state_dim(dynamics), control_dim(dynamics))
        end
        (; strategies, solution)
        # (; strategy = (OpenLoopStrategy(trajectory[1].xs, trajectory[1].us),
        #                 OpenLoopStrategy(trajectory[2].xs, trajectory[2].us)),
        # solution)
    end

    # Common obstacle
    obstacle_position = Observable([0.0, 0.0])

    # Player 1
    initial_state1 = Observable([-0.8, 0.0, 0.1, -0.1])
    goal_position1 = Observable([0.8137, -0.0279])
    θ1 = GLMakie.@lift flatten_parameters(; # θ is a flat (column) vector of parameters
        initial_state = $initial_state1,
        goal_position = $goal_position1,
        obstacle_position = $obstacle_position,
    )

    # Player 2
    initial_state2 = Observable([0.8, 0.0, -0.1, 0.1])
    goal_position2 = Observable([-0.7241, 0.0306]) 
    θ2 = GLMakie.@lift flatten_parameters(; 
        initial_state = $initial_state2,
        goal_position = $goal_position2,
        obstacle_position = $obstacle_position,
    )

    θ = GLMakie.@lift [$θ1..., $θ2...]

    println("Player 1's goal_position:", goal_position1)
    println("Player 2's goal_position:", goal_position2)

    # Main.@infiltrate

    strategy = GLMakie.@lift let
        result = get_receding_horizon_solution($θ; warmstart_solution)
        warmstart_solution = nothing
        result.strategies
    end

    # Main.@infiltrate

    figure = GLMakie.Figure()
    axis = GLMakie.Axis(figure[1, 1]; aspect = GLMakie.DataAspect(), limits = ((-1, 1), (-1, 1)))

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
        GLMakie.@lift([GLMakie.Point2f($θ1[1:2]), GLMakie.Point2f($initial_state2[1:2])]),
        markersize = 20,
        color = [:blue, :red])

    # visualize obstacle position
    GLMakie.scatter!(
        axis,
        GLMakie.@lift(GLMakie.Point2f($obstacle_position)),
        markersize = 2 * obstacle_radius * sqrt(2), # sqrt2 compensating for GLMakie bug
        markerspace = :data,
        color = (:red, 0.5),
    )

    # visualize goal positions
    GLMakie.scatter!(
        axis,
        GLMakie.@lift(GLMakie.Point2f($goal_position1)),
        markersize = 20,
        color = :green,
    )

    GLMakie.scatter!(
        axis,
        GLMakie.@lift(GLMakie.Point2f($goal_position2)),
        markersize = 20,
        color = :cyan,
    )

    # Main.@infiltrate

    # Visualize trajectories
    strategy1 = GLMakie.@lift OpenLoopStrategy($strategy[1].xs, $strategy[1].us)
    strategy2 = GLMakie.@lift OpenLoopStrategy($strategy[2].xs, $strategy[2].us)
    GLMakie.plot!(axis, strategy1)
    GLMakie.plot!(axis, strategy2)

    if record # record the simulation
        # Record for 7 seconds at a rate of 10 fps
        framerate = 10
        frames = 1:framerate * 4
        GLMakie.record(figure, filename, frames; framerate = framerate) do t
            θ1.val[1:state_dim(dynamics)] = first(strategy[]).xs[begin + 1]
            initial_state2[] = strategy[][2].xs[begin + 1]
        end
    else
        display(figure)
        while !is_stopped[]
            compute_time = @elapsed if !is_paused[]
                # Main.@infiltrate
                # Asynchronous update: mutate p1's initial state without triggering others
                θ1.val[1:state_dim(dynamics)] = first(strategy[]).xs[begin + 1]
                println("Update initial state2")
                initial_state2[] = strategy[][2].xs[begin + 1]
            end
            sleep(max(0.0, 0.1 - compute_time))
        end
        figure
    end

end

end
