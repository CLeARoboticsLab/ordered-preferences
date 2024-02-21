module SimpleMPC

export demo

using TrajectoryGamesExamples: UnicycleDynamics
using TrajectoryGamesBase:
    OpenLoopStrategy, unflatten_trajectory, state_dim, control_dim, control_bounds
using GLMakie: GLMakie, Observable
using LinearAlgebra: norm

using OrderedPreferences

function get_setup(;
    dynamics = UnicycleDynamics(),
    planning_horizon = 20,
    collision_avoidance_distance = 0.25,
)
    state_dimension = state_dim(dynamics)
    control_dimension = control_dim(dynamics)
    primal_dimension = (state_dimension + control_dimension) * planning_horizon
    goal_dimension = 2
    opponent_trajectory_dimension = 2 * planning_horizon # one position element for each time step
    parameter_dimension = state_dimension + goal_dimension + opponent_trajectory_dimension

    unflatten_parameters = function (θ)
        θ_iter = Iterators.Stateful(θ)
        initial_state = first(θ_iter, state_dimension)
        goal_position = first(θ_iter, goal_dimension)
        opponent_positions =
            reshape(first(θ_iter, opponent_trajectory_dimension), 2, :) |> eachcol |> collect
        (; initial_state, goal_position, opponent_positions)
    end

    function flatten_parameters(; initial_state, goal_position, opponent_positions)
        vcat(initial_state, goal_position, reduce(vcat, opponent_positions))
    end

    objective = function (z, θ)
        (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
        (; goal_position) = unflatten_parameters(θ)
        sum(sum(u .^ 2) for u in us)
    end

    equality_constraints = function (z, θ)
        (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
        (; initial_state) = unflatten_parameters(θ)
        initial_state_constraint = xs[1] - initial_state
        dynamics_constraints = mapreduce(vcat, 2:length(xs)) do k
            xs[k] - dynamics(xs[k - 1], us[k - 1], k)
        end
        vcat(initial_state_constraint, dynamics_constraints)
    end

    function inequality_constraints(z, θ)
        (; lb, ub) = control_bounds(dynamics)
        lb_mask = findall(!isinf, lb)
        ub_mask = findall(!isinf, ub)
        (; us) = unflatten_trajectory(z, state_dimension, control_dimension)
        mapreduce(vcat, us) do u
            vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
        end
    end

    prioritized_inequality_constraints = [
        # most important: avoid collisions with opponent at every time step
        function (z, θ)
            (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
            (; opponent_positions) = unflatten_parameters(θ)
            mapreduce(vcat, 2:length(xs)) do k
                sum((xs[k][1:2] - opponent_positions[k]) .^ 2) - collision_avoidance_distance^2
            end
        end,

        # limit acceleration and don't go too fast, stay within the playing field
        function (z, θ)
            (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
            mapreduce(vcat, 1:length(xs)) do k
                px, py, v, θ = xs[k]
                a, ω = us[k]
                lateral_acceleration = v * ω
                longitudinal_acceleration = a
                acceleration_constarint =
                    0.5 - (lateral_acceleration^2 + longitudinal_acceleration^2)
                velocity_constraint = vcat(v + 1.0, -v + 1.0)
                position_constraints = vcat(px + 5.0, -px + 5.0, py + 5.0, -py + 5.0)
                vcat(acceleration_constarint, velocity_constraint, position_constraints)
            end
        end,

        # reach the goal
        function (z, θ)
            (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
            (; goal_position) = unflatten_parameters(θ)
            goal_deviation = xs[end][1:2] .- goal_position
            [
                goal_deviation .+ 0.01
                -goal_deviation .+ 0.01
            ]
        end,
    ]

    problem = ParametricOrderedPreferencesProblem(;
        objective,
        equality_constraints,
        inequality_constraints,
        prioritized_inequality_constraints,
        primal_dimension,
        parameter_dimension,
    )

    #function compute_optimized_trajectory(
    #    initial_state,
    #    goal_position,
    #    opponent_positions;
    #    warmstart_solution = nothing,
    #)
    #    θ = flatten_parameters(;
    #        initial_state = initial_state,
    #        goal_position = goal_position,
    #        opponent_positions = opponent_positions,
    #    )
    #    solution = solve(problem, θ; warmstart_solution)
    #    unflatten_trajectory(solution.primals, state_dimension, control_dimension)
    #end
    #
    (; problem, flatten_parameters, unflatten_parameters)
end

function demo(; verbose = false, paused = false, record = false, filename = "SimpleMPC_with_two_players.mp4")
    dynamics = UnicycleDynamics(; control_bounds = (; lb = [-1.0, -1.0], ub = [1.0, 1.0]))
    collision_avoidance_distance = 0.30

    # For computing initial trajectory
    planning_horizon = 20
    (; problem, flatten_parameters, unflatten_parameters) = get_setup(; dynamics, planning_horizon, collision_avoidance_distance)

    # TODO: Now:
    #
    # 1. construct the `best_response_map` for each player as a callable:
    #   `best_response_map(parameters::Vector{Float64}, initial_guess::Vector{Float64})::Vector{Float64}`
    # 2. call `solve_nash` with the `best_response_map`s and an initial guess for the trajectory
    #
    # This whole logic will take the role of `get_receding_horizon_solution`.

    warmstart_solution = nothing

    function get_receding_horizon_solution(problem, θ; warmstart_solution)
        solution = solve(problem, θ; warmstart_solution)
        trajectory =
            unflatten_trajectory(solution.primals, state_dim(dynamics), control_dim(dynamics))
        (; strategy = OpenLoopStrategy(trajectory.xs, trajectory.us), solution) # return NamedTuple
    end

    function get_random_point_within_ball(center, radius)
        # Check center is Tuple
        @assert length(center) == 2 "Center must be a 2-element vector [x, y]"
        x_coord, y_coord = center

        # Generate random angle in radians
        angle = 2π * rand()

        # Generate random distance within the specificed radius
        r = radius * sqrt(rand())

        # Calculate new x and y coordinates
        x = x_coord + r * cos(angle)
        y = y_coord + r * sin(angle)

        [x, y]
    end
    ##
    # Player 1
    initial_state1 = Observable([-0.8, 0.0, 0.0, 0.0])
    goal_position1 = Observable([0.8137, -0.0279]) #Observable(get_random_point_within_ball((0.8, 0.0), 0.1))
    opponent_position1 = Observable(zeros(2, planning_horizon) |> eachcol |> collect) # initially zeros 
    θ1 = GLMakie.@lift flatten_parameters(; # θ is a flat (column) vector of parameters
        initial_state = $initial_state1,
        goal_position = $goal_position1,
        opponent_positions = $opponent_position1,
    )

    # Player 2
    initial_state2 = Observable([0.8, 0.0, 0.0, 0.0])
    goal_position2 = Observable([-0.7241, 0.0306]) #Observable(get_random_point_within_ball((-0.8, 0.0), 0.1))
    opponent_position2 = Observable(zeros(2, planning_horizon) |> eachcol |> collect)
    θ2 = GLMakie.@lift flatten_parameters(; 
        initial_state = $initial_state2,
        goal_position = $goal_position2,
        opponent_positions = $opponent_position2,
    )

    println("Player 1's goal_position:", goal_position1)
    println("Player 2's goal_position:", goal_position2)

    function best_response_map(parameters::Vector{Float64}, initial_guess::Union{NamedTuple, Nothing}, opponent_positions::Union{Vector{Float64}, Nothing}) 
        # Update player's opponent_positions in θ
        if !isnothing(opponent_positions)
            parameters[end - 2*planning_horizon + 1: end] = opponent_positions 
        end
        solution = solve(problem, parameters; warmstart_solution = initial_guess, warmstart_strategy = :parallel) # does better with parallel?
        trajectory = unflatten_trajectory(solution.primals, state_dim(dynamics), control_dim(dynamics))
        (; strategy = OpenLoopStrategy(trajectory.xs, trajectory.us), solution)
    end

    # Create best_response_maps
    best_response_maps = GLMakie.@lift let 
        [
        (initial_guess, opponent_positions) -> best_response_map($θ1, initial_guess, opponent_positions),
        (initial_guess, opponent_positions) -> best_response_map($θ2, initial_guess, opponent_positions)]
    end

    initial_trajectory_guesses = Union{Vector{Vector{Float64}}, Nothing}[nothing for _ in 1:length(best_response_maps[])]

    # Solve Nash
    trajectories = GLMakie.@lift let 
        solve_nash!($best_response_maps, initial_trajectory_guesses; verbose = verbose)
    end

    # Visualize
    figure = GLMakie.Figure()
    axis = GLMakie.Axis(figure[1, 1]; aspect = GLMakie.DataAspect(), limits = ((-1, 1), (-1, 1)))

    # controlling the goal_position1 with the RIGHT mouse button
    is_goal_position1_locked = GLMakie.Observable(true)
    GLMakie.on(GLMakie.events(figure).mouseposition, priority = 0) do _
        if !is_goal_position1_locked[]
            goal_position1[] = GLMakie.mouseposition(axis.scene) # Control player 1
        end
        GLMakie.Consume(false)
    end
    GLMakie.on(GLMakie.events(figure).mousebutton, priority = 0) do event
        if event.button == GLMakie.Mouse.right
            if event.action == GLMakie.Mouse.press
                is_goal_position1_locked[] = !is_goal_position1_locked[]
            end
        end
        GLMakie.Consume(false)
    end

        # # controlling the goal_position2 with the LEFT mouse button
        # is_goal_position2_locked = GLMakie.Observable(true)
        # GLMakie.on(GLMakie.events(figure).mouseposition, priority = 1) do _
        #     if !is_goal_position2_locked[]
        #         goal_position2[] = GLMakie.mouseposition(axis.scene) # Control player 2
        #     end
        #     GLMakie.Consume(false)
        # end
        # GLMakie.on(GLMakie.events(figure).mousebutton, priority = 1) do event
        #     if event.button == GLMakie.Mouse.left
        #         if event.action == GLMakie.Mouse.press
        #             is_goal_position2_locked[] = !is_goal_position2_locked[]
        #         end
        #     end
        #     GLMakie.Consume(false)
        # end

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

    # visualize initial states
    GLMakie.scatter!(
        axis,
        GLMakie.@lift(GLMakie.Point2f($initial_state1[1:2])),
        markersize = 20,
        color = :blue,
    )
    GLMakie.scatter!(
        axis,
        GLMakie.@lift(GLMakie.Point2f($initial_state2[1:2])),
        markersize = 20,
        color = :red,
    )

    # # visualize obstacle position
    # GLMakie.scatter!(
    #     axis,
    #     GLMakie.@lift(GLMakie.Point2f($obstacle_position)),
    #     markersize = 2 * collision_avoidance_distance * sqrt(2), # sqrt2 compensating for GLMakie bug
    #     markerspace = :data,
    #     color = (:red, 0.5),
    # )

    # visualize goal position
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

    # visualize trajectories
    strategy1 = GLMakie.@lift OpenLoopStrategy($trajectories[1], nothing)
    strategy2 = GLMakie.@lift OpenLoopStrategy($trajectories[2], nothing)
    GLMakie.plot!(axis, strategy1)
    GLMakie.plot!(axis, strategy2)

    if record # record the simulation
        # Record for 7 seconds at a rate of 10 fps
        framerate = 10
        frames = 1:framerate * 7
        GLMakie.record(figure, filename, frames; framerate = framerate) do t
            initial_state1[] = strategy1[].xs[begin + 1]
            initial_state2[] = strategy2[].xs[begin + 1]
        end
    else
        display(figure)
        while !is_stopped[]
            compute_time = @elapsed if !is_paused[]
                println("Update initial state1") # TODO: Asynchronous update
                initial_state1[] = strategy1[].xs[begin + 1]
                initial_state2[] = strategy2[].xs[begin + 1]
            end
            sleep(max(0.0, 0.1 - compute_time))
        end
        figure
    end
    
end

end
