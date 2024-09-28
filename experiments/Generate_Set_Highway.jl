module Generate_Set_Highway

using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using TrajectoryGamesBase:
    OpenLoopStrategy, unflatten_trajectory, state_dim, control_dim, control_bounds
using BlockArrays
using JLD2

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
            [
                sum((xs[1][k][1:2] - xs[2][k][1:2]) .^ 2) - collision_avoidance^2
                sum((xs[1][k][1:2] - xs[3][k][1:2]) .^ 2) - collision_avoidance^2
                sum((xs[2][k][1:2] - xs[3][k][1:2]) .^ 2) - collision_avoidance^2
            ]
        end
    end

    shared_equality_dimension = length(shared_equality_constraints(dummy_primals, dummy_parameters))
    shared_inequality_dimension = length(shared_inequality_constraints(dummy_primals, dummy_parameters))

    problem = ParametricGameClassifier(;
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
    )

    (; problem, flatten_parameters, unflatten_parameters)
end

function demo(; verbose = false, paused = false, record = false, filename = "N_player_GOOP_v1.mp4")

    num_players = 3
    dynamics = planar_double_integrator(; control_bounds = (; lb = [-2.0, -2.0], ub = [2.0, 2.0])) # x := (px, py, vx, vy) and u := (ax, ay).
    planning_horizon = 5
    collision_avoidance = 0.2

    (; problem, flatten_parameters) = get_setup(num_players; dynamics, planning_horizon, collision_avoidance)

    warmstart_solution = nothing

    dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
    primal_dimension = dynamics_dimension * planning_horizon

    function get_receding_horizon_solution(θ; warmstart_solution)
        solution = classify_game(problem, θ; initial_guess = warmstart_solution, verbose, return_primals = true)

        return solution # (; primals, variables = z, objectives, status, info)
    end

    function generate_samples(num_samples = 10, range = 0.05)
        samples = []
        ii, jj = 1, 1
        obstacle_position = [0.5, 0.5] # placeholder
        warmstart_solution = nothing

        while ii + jj <= num_samples
            # Sample a random number between x and y = x + rand()*(y-x)
            initial_state1 = [-0.3, -0.2 + rand()*0.4, 0.4 + rand() * (0.8 - 0.4), 0.0]
            initial_state2 = [0.1, -0.2 + rand()*0.4, 0.2, 0.0]
            initial_state3 = [0.4, -0.2 + rand()*0.4, 0.2, 0.0]

            # Goal positions
            goal_position1 = [0.9, 0.0]
            goal_position2 = [0.9, 0.0]
            goal_position3 = [0.9, 0.0]

            # Parameters
            θ1 = flatten_parameters(; # θ is a flat (column) vector of parameters
            initial_state = initial_state1,
            goal_position = goal_position1,
            obstacle_position = obstacle_position)

            θ2 = flatten_parameters(;
            initial_state = initial_state2,
            goal_position = goal_position2,
            obstacle_position = obstacle_position)

            θ3 = flatten_parameters(;
            initial_state = initial_state3,
            goal_position = goal_position3,
            obstacle_position = obstacle_position,)
            θ = [θ1..., θ2..., θ3...]

            # Classify problem instance
            solution = get_receding_horizon_solution(θ; warmstart_solution)

            # Define states_dict for saving
            states_dict = Dict(
                "initial_state1" => initial_state1,
                "initial_state2" => initial_state2,
                "initial_state3" => initial_state3
            )

            # 1. If no sol, this is infeasible problem and we drop the instance
            if string(solution.status) != "MCP_Solved"
                println("Infeasible problem...moving on to the next problem")
                continue

            # 2. If sol with positive objective, this is a relaxably feasible problem and goes to "Relaxably Feasible Set" 
            elseif any(x -> x > 0, solution.objectives)
                println("Relaxably feasible problem...saving the instance #$ii")
                file_name = "./data/relaxably_feasible/rfp_$ii.jld2"
                ii += 1

            # 3. If sol with zero objective, this is a trivially feasible problem and goes to "Trivially Feasible Set"
            elseif all(x -> isapprox(x, 0.0; atol=1e-4), solution.objectives)
                println("Trivially feasible problem...saving the instance #$jj")
                file_name = "./data/trivially_feasible/tfp_$jj.jld2"
                jj += 1

            else
                println("Inspection needed for this problem...")
            end
            println(solution.objectives)
            JLD2.save_object(file_name, states_dict)
            # Main.@infiltrate
        end
        samples
    end

    samples = generate_samples(10)

    for sample in samples
        println(sample)
    end
end

end
