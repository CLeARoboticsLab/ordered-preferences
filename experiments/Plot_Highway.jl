using CairoMakie
using Makie.GeometryBasics  # Required for defining polygons
using JLD2
using TrajectoryGamesBase: OpenLoopStrategy

function create_map(; n_time = 7)
    # Create a figure (adjust resolution as needed)
    fig = Figure(size = (500,1000))
    
    # Define time steps
    lane_ids = 1:n_time
    lanes = Dict{Int, Axis}()
    
    # Set a lane "height" for drawing (adjust as needed)
    lane_height = 0.5
    top_edge = lane_height / 2
    bottom_edge = -lane_height / 2
    
    for (i, lane) in enumerate(lane_ids)
        # Create an axis for this lane in row i, column 1 of the figure grid.
        lane_ax = Axis(fig[i, 1], aspect = DataAspect(),
                       xgridvisible = false, ygridvisible = false)
        hidedecorations!(lane_ax)
        hidespines!(lane_ax)
        lanes[lane] = lane_ax
        
        # Draw the lane boundaries:
        # - Solid top and bottom boundaries for every lane.
        lines!(lane_ax, [(-1, top_edge), (1, top_edge)], color = :black)
        lines!(lane_ax, [(-1, bottom_edge), (1, bottom_edge)], color = :black)
        lines!(lane_ax, [(-1, 0), (1, 0)], color = :black, linestyle = :dash)
        
    end
    
    return fig, lanes
end

function create_data_fig()
    # Visualize vertical speed, horizontal speed, ...
    fig = Figure(size = (700, 700))
    return fig
end

function draw_highway_closed_loop!(fig1, lanes, fig2; n_time = 7)
    closed_loop_goop_strategy1 = Vector{Float64}[]
    closed_loop_goop_strategy2 = Vector{Float64}[]
    closed_loop_goop_strategy3 = Vector{Float64}[]
    # 1. plot GOOP solution    
    for ii in 1:n_time
        data = load_object("data/Highway_closed_loop/GOOP_solution/Highway_39_w1_T$(ii)_sol.jld2")

        initial_state1 = data["strategy1"].xs[1][1:2]
        initial_state2 = data["strategy2"].xs[1][1:2]
        initial_state3 = data["strategy3"].xs[1][1:2]
        goal_position = [0.9, 0.0]

        # add text
        text!(lanes[ii], 0.05, 0.35, text = "T=$ii [s]", font=:bold, align=(:left, :top), offset=(4, -2), space=:relative, fontsize = 18)

        # Visualize initial states
        scatter!(lanes[ii], Point2f(initial_state1), color = :blue, markersize = 20)
        scatter!(lanes[ii], Point2f(initial_state2), color = :red, markersize = 20)
        scatter!(lanes[ii], Point2f(initial_state3), color = :green, markersize = 20)

        # Visualize goal position
        scatter!(lanes[ii], Point2f(goal_position), color = :grey, marker = :star5, markersize = 20)

        # Store the strategies for closed-loop visualization
        goop_strategy1 = data["strategy1"]
        goop_strategy2 = data["strategy2"]
        goop_strategy3 = data["strategy3"]
        push!(closed_loop_goop_strategy1, first(goop_strategy1.xs))
        push!(closed_loop_goop_strategy2, first(goop_strategy2.xs))
        push!(closed_loop_goop_strategy3, first(goop_strategy3.xs))

        # Visualize trajectories
        goop_strategy1_xs = [vcat(v[2], v[1], v[3:end]) for v in goop_strategy1.xs] # This will change goop_strategy1.xs
        goop_strategy2_xs = [vcat(v[2], v[1], v[3:end]) for v in goop_strategy2.xs]
        goop_strategy3_xs = [vcat(v[2], v[1], v[3:end]) for v in goop_strategy3.xs]
        # NOTE: The previous code is a workaround to plot the trajectories correctly.
        goop_strategy1 = OpenLoopStrategy(goop_strategy1_xs, goop_strategy1.us)
        goop_strategy2 = OpenLoopStrategy(goop_strategy2_xs, goop_strategy2.us)
        goop_strategy3 = OpenLoopStrategy(goop_strategy3_xs, goop_strategy3.us)
        plot!(lanes[ii], goop_strategy1, color = :blue, markersize = 10)
        plot!(lanes[ii], goop_strategy2, color = :red, markersize = 10)
        plot!(lanes[ii], goop_strategy3, color = :green, markersize = 10)

        # Visualize past trajectories (closed-loop)
        if ii > 1 
            blue_trajectory_xs = [data[1] for data in closed_loop_goop_strategy1]
            blue_trajectory_ys = [data[2] for data in closed_loop_goop_strategy1]
            red_trajectory_xs = [data[1] for data in closed_loop_goop_strategy2]
            red_trajectory_ys = [data[2] for data in closed_loop_goop_strategy2]
            green_trajectory_xs = [data[1] for data in closed_loop_goop_strategy3]
            green_trajectory_ys = [data[2] for data in closed_loop_goop_strategy3]
            # Draw the past trajectories
            for j in 1:ii-1
                scatterlines!(
                    lanes[ii],
                    [blue_trajectory_xs[j], blue_trajectory_xs[j+1]],
                    [blue_trajectory_ys[j], blue_trajectory_ys[j+1]],
                    color = (:blue, 0.3),
                    linewidth = 2,
                    linestyle = (:dash, :dense)
                )
                scatterlines!(
                    lanes[ii],
                    [red_trajectory_xs[j], red_trajectory_xs[j+1]],
                    [red_trajectory_ys[j], red_trajectory_ys[j+1]],
                    color = (:red, 0.3),
                    linewidth = 2,
                    linestyle = (:dash, :dense)
                )
                scatterlines!(
                    lanes[ii],
                    [green_trajectory_xs[j], green_trajectory_xs[j+1]],
                    [green_trajectory_ys[j], green_trajectory_ys[j+1]],
                    color = (:green, 0.3),
                    linewidth = 2,
                    linestyle = (:dash, :dense)
                )
            end
        end
    end

    Main.@infiltrate
    # TODO: Plot closed-loop data like speed, distance to goal
    # TODO: do the same for baselines


    return fig1, fig2
end

fig1, lanes = create_map(n_time = 7)
fig2 = create_data_fig() # visualize speed, distance, etc.
fig1, fig2 = draw_highway_closed_loop!(fig1, lanes, fig2; n_time = 7)
display(fig1)