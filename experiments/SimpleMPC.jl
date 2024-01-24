module SimpleMPC
export demo
using TrajectoryGamesExamples: UnicycleDynamics
using TrajectoryGamesBase:
    OpenLoopStrategy, unflatten_trajectory, state_dim, control_dim, control_bounds
using GLMakie: GLMakie, Observable

using OrderedPreferences

function get_setup(; dynamics = UnicycleDynamics(), planning_horizon = 20, obstacle_radius = 0.25, collision_dist = 0.02, others = nothing)
    state_dimension = state_dim(dynamics)
    control_dimension = control_dim(dynamics)
    primal_dimension = (state_dimension + control_dimension) * planning_horizon
    parameter_dimension = state_dimension + 4

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
        # most important: obstacle avoidance
        function (z, θ)
            (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
            (; obstacle_position) = unflatten_parameters(θ)
            mapreduce(vcat, 2:length(xs)) do k
                sum((xs[k][1:2] - obstacle_position) .^ 2) - obstacle_radius^2
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

    # Add collision avoidance constraint (now most important) if other players exist
    if !isnothing(others) # TODO: Expand to include more than 2 players
        #Main.@infiltrate
        pushfirst!(prioritized_inequality_constraints, function (z, θ)
            (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
            mapreduce(vcat, 2:length(xs)) do k
                sum((xs[k][1:2] - others[k][1:2]) .^ 2) - collision_dist^2
            end
        end)
    end
    
    problem = ParametricOrderedPreferencesProblem(;
        objective,
        equality_constraints,
        inequality_constraints,
        prioritized_inequality_constraints,
        primal_dimension,
        parameter_dimension,
    )
    
    (; problem, flatten_parameters, unflatten_parameters)
end

function demo(; paused = false)
    dynamics = UnicycleDynamics(; control_bounds = (; lb = [-1.0, -1.0], ub = [1.0, 1.0]))
    obstacle_radius = 0.25
    collision_dist = 0.02

    # For computing initial trajectory
    (; problem, flatten_parameters) = get_setup(; dynamics, obstacle_radius, collision_dist)

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

    obstacle_position = Observable([-0.5, 0.0])

    # Player 1
    initial_state1 = Observable([-0.25, -0.5, 0.0, 0.0])
    goal_position1 = Observable(get_random_point_within_ball((-0.5, 0.5), 0.1))
    θ1 = GLMakie.@lift flatten_parameters(; initial_state = $initial_state1, goal_position = $goal_position1, obstacle_position = $obstacle_position)

    # Player 2
    initial_state2 = Observable([0.75, -0.5, 0.0, 0.0])
    goal_position2 = Observable(get_random_point_within_ball((-0.5, -0.5), 0.1))
    θ2 = GLMakie.@lift flatten_parameters(; initial_state = $initial_state2, goal_position = $goal_position2, obstacle_position = $obstacle_position)

    println("Player 1's goal_position:", goal_position1)
    println("Player 2's goal_position:", goal_position2)

#   function get_iterated_best_response()
    # 1. Initialize trajectory for Player 2 
    strategy2 = GLMakie.@lift let
         result = get_receding_horizon_solution(problem, $θ2; warmstart_solution)
         warmstart_solution = result.solution
         result.strategy
    end


    Main.@infiltrate

    # 2. Solve for each player's best response
    # Player 1 goes first, then Player 2 responses based on Player 1's trajectory
    # Include collision avoidance priority constraint
    (; problem) = get_setup(; dynamics, obstacle_radius, collision_dist, others = strategy2.val.xs)
    strategy1 = GLMakie.@lift let 
        result = get_receding_horizon_solution(problem, $θ1; warmstart_solution)
        result.strategy # P1's best response
    end

    Main.@infiltrate

    (; problem) = get_setup(; dynamics, obstacle_radius, collision_dist, others = strategy1.val.xs)
    strategy2 = GLMakie.@lift let 
        result = get_receding_horizon_solution(problem, $θ2; warmstart_solution)
        result.strategy # P2's best response
    end

#    end

    Main.@infiltrate


    # initial_state = Observable(zeros(state_dim(dynamics)))
    # goal_position = Observable([0.5, 0.5])
    # obstacle_position = Observable([-0.5, 0.0])

    # θ = GLMakie.@lift flatten_parameters(;
    #     initial_state = $initial_state,
    #     goal_position = $goal_position,
    #     obstacle_position = $obstacle_position,
    # )

    # strategy = GLMakie.@lift let
    #     result = get_receding_horizon_solution($θ; warmstart_solution)
    #     warmstart_solution = result.solution
    #     result.strategy
    # end

    figure = GLMakie.Figure()
    axis = GLMakie.Axis(figure[1, 1]; aspect = GLMakie.DataAspect(), limits = ((-1, 1), (-1, 1)))

    # mouse interaction
    # controlling the obstacle with the left mouse button
    is_obstacle_position_locked = GLMakie.Observable(true)
    GLMakie.on(GLMakie.events(figure).mouseposition, priority = 0) do _
        if !is_obstacle_position_locked[]
            obstacle_position[] = GLMakie.mouseposition(axis.scene)
        end
        GLMakie.Consume(false)
    end
    GLMakie.on(GLMakie.events(figure).mousebutton, priority = 0) do event
        if event.button == GLMakie.Mouse.left
            if event.action == GLMakie.Mouse.press
                is_obstacle_position_locked[] = !is_obstacle_position_locked[]
            end
        end
        GLMakie.Consume(false)
    end
    # controlling the goal position with the right mouse button
    is_goal_position_locked = GLMakie.Observable(true)
    GLMakie.on(GLMakie.events(figure).mouseposition, priority = 1) do _
        if !is_goal_position_locked[]
            goal_position[] = GLMakie.mouseposition(axis.scene)
        end
        GLMakie.Consume(false)
    end
    GLMakie.on(GLMakie.events(figure).mousebutton, priority = 1) do event
        if event.button == GLMakie.Mouse.right
            if event.action == GLMakie.Mouse.press
                is_goal_position_locked[] = !is_goal_position_locked[]
            end
        end
        GLMakie.Consume(false)
    end

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

    # visualize initial state
    GLMakie.scatter!(axis, GLMakie.@lift(GLMakie.Point2f($initial_state1[1:2])), markersize = 20, color = :blue)
    GLMakie.scatter!(axis, GLMakie.@lift(GLMakie.Point2f($initial_state2[1:2])), markersize = 20, color = :red)

    # visualize obstacle position
    GLMakie.scatter!(
        axis,
        GLMakie.@lift(GLMakie.Point2f($obstacle_position)),
        markersize = 2 * obstacle_radius * sqrt(2), # sqrt2 compensating for GLMakie bug
        markerspace = :data,
        color = (:red, 0.5),
    )

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

    GLMakie.plot!(axis, strategy1, strategy2)

    display(figure)

    while !is_stopped[]
        compute_time = @elapsed if !is_paused[]
            initial_state[] = strategy[].xs[begin + 1]
        end
        sleep(max(0.0, 0.1 - compute_time))
    end

    figure
end

end
