module SingleAgent_KKT

using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using TrajectoryGamesBase:
    OpenLoopStrategy, unflatten_trajectory, state_dim, control_dim, control_bounds
using GLMakie: GLMakie, Observable

using OrderedPreferences

function get_setup(; dynamics = UnicycleDynamics, planning_horizon = 20, obstacle_radius = 0.25, relaxation_mode = :standard)
    state_dimension = state_dim(dynamics)
    control_dimension = control_dim(dynamics)
    primal_dimension = (state_dimension + control_dimension) * planning_horizon
    parameter_dimension = state_dimension + 4 # (state, goal, obstacle)

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
        (; xs, us) = unflatten_trajectory(z[1:primal_dimension], state_dimension, control_dimension)
        (; goal_position) = unflatten_parameters(θ)

        sum(sum(u .^ 2) for u in us) #+ 10*sum((xs[end][1:2] .- goal_position) .^ 2) # penalty term for reaching goal
    end

    equality_constraints = function (z, θ)
        (; xs, us) = unflatten_trajectory(z[1:primal_dimension], state_dimension, control_dimension)
        (; initial_state) = unflatten_parameters(θ)
        initial_state_constraint = xs[1] - initial_state
        dynamics_constraints = mapreduce(vcat, 2:length(xs)) do k
            xs[k] - dynamics(xs[k - 1], us[k - 1], k)
        end
        vcat(initial_state_constraint, dynamics_constraints)
    end
    equality_dimension = length(equality_constraints(zeros(primal_dimension), zeros(parameter_dimension)))
   
    function inequality_constraints(z, θ)
        (; lb, ub) = control_bounds(dynamics)
        lb_mask = findall(!isinf, lb)
        ub_mask = findall(!isinf, ub)
        (; xs, us) = unflatten_trajectory(z[1:primal_dimension], state_dimension, control_dimension)
        vcat(
            # control bounds (box)
            mapreduce(vcat, us) do u
                 vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
            end,
            # control bounds (||u||_2 ≤ 1)
            # mapreduce(vcat, us) do u
            #     -norm(u) + 1.0
            # end,
            # limit acceleration and don't go too fast, stay within the playing field (for planar_double_integrator)
            mapreduce(vcat, 1:length(xs)) do k
                px, py, vx, vy = xs[k]
                position_constraints = vcat(px + 1.0, -px + 1.0, py + 1.0, -py + 1.0)
                vcat(position_constraints)
            end
            # limit acceleration and don't go too fast, stay within the playing field
            # mapreduce(vcat, 1:length(xs)) do k
            #     px, py, v, θ = xs[k]
            #     a, ω = us[k]

            #     lateral_acceleration = v * ω
            #     longitudinal_acceleration = a
            #     acceleration_constarint =
            #         0.5 - (lateral_acceleration^2 + longitudinal_acceleration^2)
            #     velocity_constraint = vcat(v + 1.0, -v + 1.0)
            #     position_constraints = vcat(px + 1.0, -px + 1.0, py + 1.0, -py + 1.0)
            #     vcat(acceleration_constarint, velocity_constraint, position_constraints)
            # end
        )
    end
    inequality_dimension = length(inequality_constraints(zeros(primal_dimension), zeros(parameter_dimension)))

    prioritized_preferences = [

        # most important: obstacle avoidance
        function (z, θ)
            (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
            (; obstacle_position) = unflatten_parameters(θ)
            mapreduce(vcat, 2:length(xs)) do k
                sum((xs[k][1:2] - obstacle_position) .^ 2) - obstacle_radius^2
            end
        end,

        # # simplified 
        # function (z, θ)
        #     (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
        #     # p_x ≥ 0.0
        #     mapreduce(vcat, 2:length(xs)) do k
        #         xs[k][1]
        #     end
        # end,
        
        # simplified with p_y[end] ≤ 0.0
        function (z, θ)
            (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
            -xs[end][2]
        end,

        # # simplified with p_x[end] ≥ 0.0
        # function (z, θ)
        #     (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
        #     xs[end][1]
        # end,

        # # reach the goal. Instead, try: -sum((xs[end][1:2] - goal_position) .^ 2) + 0.01^2 (not prioritized constraint)
        # function (z, θ)
        #     (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
        #     (; goal_position) = unflatten_parameters(θ)
        #     goal_deviation = xs[end][1:2] .- goal_position
        #     [
        #         goal_deviation .+ 0.01
        #         -goal_deviation .+ 0.01
        #     ]
        #     # -sum((xs[end][1:2] - goal_position) .^ 2)
        # end,

        # # simplified 
        # function (z, θ)
        #     (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
        #     # p_y ≤ 0.0
        #     mapreduce(vcat, 2:length(xs)) do k
        #         -xs[k][2]
        #     end
        # end,
    ]

    # Specify priortized constraint
    is_prioritized_constraint = [true, true, true]

    problem = ParametricOrderedPreferencesMPCC(; 
        objective,
        equality_constraints,
        inequality_constraints,
        prioritized_preferences,
        is_prioritized_constraint,
        primal_dimension,
        parameter_dimension,
        equality_dimension,
        inequality_dimension,
        relaxation_mode,
    )

    (; problem, flatten_parameters, unflatten_parameters)
end

function demo(; verbose = false, paused = false, record = false, filename = "Single_agent_KKT.mp4")
    # Algorithm setting
    ϵ = 1.1
    κ = 0.1
    max_iterations = 16
    tolerance = 1e-3
    relaxation_mode = :standard

    # dynamics = UnicycleDynamics(; control_bounds = (; lb = 10*[-1.0, -1.0], ub = 10*[1.0, 1.0])) # x := (px, py, v, θ) and u := (a, ω). Need to give initial velocity
    dynamics = planar_double_integrator(; control_bounds = (; lb = [-1.0, -1.0], ub = [1.0, 1.0])) # x := (px, py, vx, vy) and u := (ax, ay).
    planning_horizon = 10
    obstacle_radius = 0.25
    (; problem, flatten_parameters) = get_setup(; dynamics, planning_horizon, obstacle_radius, relaxation_mode)

    warmstart_solution = nothing

    dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
    primal_dimension = dynamics_dimension * planning_horizon

    function get_receding_horizon_solution(θ; warmstart_solution)
        (; relaxation, solution, residual) =
            solve_relaxed_pop(problem, warmstart_solution, θ; ϵ, κ, max_iterations, tolerance, verbose)
        println("residual: ", residual)
        println("relaxation: ", relaxation)

        trajectory =
            unflatten_trajectory(solution[end].primals[1:primal_dimension], state_dim(dynamics), control_dim(dynamics))
        (; strategy = OpenLoopStrategy(trajectory.xs, trajectory.us), solution)
    end

    initial_state = Observable([-1.0, 1.0, 1.0, 0.0]) #zeros(state_dim(dynamics))
    goal_position = Observable([-0.2, 0.1])
    obstacle_position = Observable([0.25, 0.0])

    θ = GLMakie.@lift flatten_parameters(;
        initial_state = $initial_state,
        goal_position = $goal_position,
        obstacle_position = $obstacle_position,
    )

    strategy = GLMakie.@lift let
        result = get_receding_horizon_solution($θ; warmstart_solution)
        warmstart_solution = result.solution[end].variables
        # Shift warmstart_solution by 1 time step
        warmstart_solution = vcat(warmstart_solution[dynamics_dimension + 1:end], zeros(dynamics_dimension)) 
        result.strategy
    end

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
    GLMakie.scatter!(axis, GLMakie.@lift(GLMakie.Point2f($initial_state[1:2])), markersize = 20)

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
        GLMakie.@lift(GLMakie.Point2f($goal_position)),
        markersize = 20,
        color = :green,
    )

    GLMakie.plot!(axis, strategy)

    if record # record the simulation
        # Record for 7 seconds at a rate of 10 fps
        framerate = 10
        frames = 1:framerate * 7
        GLMakie.record(figure, filename, frames; framerate = framerate) do t
            initial_state[] =  strategy[].xs[begin + 1]
        end
    else
        display(figure)
        while !is_stopped[]
            compute_time = @elapsed if !is_paused[]
                Main.@infiltrate
                initial_state[] =  strategy[].xs[begin + 1]
            end
            sleep(max(0.0, 0.1 - compute_time))
        end
        figure
    end

end

end
