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

function draw_highway_closed_loop!(fig1, lanes, fig2; n_time = 7, method = "GOOP")
    closed_loop_goop_strategy1 = Vector{Float64}[]
    closed_loop_goop_strategy2 = Vector{Float64}[]
    closed_loop_goop_strategy3 = Vector{Float64}[]

    scale_factor = 27.778 # Scale factor for converting to real-world units (m/s)

    # 1. plot GOOP solution    
    for ii in 1:n_time
        if method == "GOOP"
            data = load_object("data/Highway_closed_loop/$(method)_solution/Highway_39_w1_T$(ii)_sol.jld2")
        else
            data = load_object("data/Highway_closed_loop/$(method)_solution/rfp_39_B2_T$(ii)_sol.jld2")
        end

        initial_state1 = data["strategy1"].xs[1][1:2]
        initial_state2 = data["strategy2"].xs[1][1:2]
        initial_state3 = data["strategy3"].xs[1][1:2]
        goal_position = [0.9, 0.0]

        # add text
        text!(lanes[ii], 0.05, 0.35, text = "t=$(ii-1) [s]", font=:bold, align=(:left, :top), offset=(4, -2), space=:relative, fontsize = 18)

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
            if ii == n_time
                #Plot closed-loop data like speed, distance to goal
                closed_loop_horizon =  n_time
                blue_trajectory_xs = [data[1] for data in closed_loop_goop_strategy1] .* scale_factor
                blue_trajectory_ys = [data[2] for data in closed_loop_goop_strategy1] .* scale_factor
                blue_trajectory_vxs = [data[3] for data in closed_loop_goop_strategy1] .* scale_factor 
                blue_trajectory_vys = [data[4] for data in closed_loop_goop_strategy1] .* scale_factor
                red_trajectory_xs = [data[1] for data in closed_loop_goop_strategy2] .* scale_factor
                red_trajectory_ys = [data[2] for data in closed_loop_goop_strategy2] .* scale_factor
                red_trajectory_vxs = [data[3] for data in closed_loop_goop_strategy2] .* scale_factor 
                red_trajectory_vys = [data[4] for data in closed_loop_goop_strategy2] .* scale_factor 
                green_trajectory_xs = [data[1] for data in closed_loop_goop_strategy3] .* scale_factor
                green_trajectory_ys = [data[2] for data in closed_loop_goop_strategy3] .* scale_factor
                green_trajectory_vxs = [data[3] for data in closed_loop_goop_strategy3] .* scale_factor
                green_trajectory_vys = [data[4] for data in closed_loop_goop_strategy3] .* scale_factor
                # 1a. Visualize horizontal speed
                axis2 = Axis(fig2[1, 1]; xlabel = "time step", ylabel = "longitudinal speed [m/s]", xgridvisible = false, ygridvisible = false)
                scatterlines!(axis2, 0:closed_loop_horizon-1, blue_trajectory_vxs, label = "Vehicle 1", color = :blue)
                scatterlines!(axis2, 0:closed_loop_horizon-1, red_trajectory_vxs, label = "Vehicle 2", color = :red)
                scatterlines!(axis2, 0:closed_loop_horizon-1, green_trajectory_vxs, label = "Vehicle 3", color = :green)
                lines!(axis2, 0:closed_loop_horizon-1, [0.2 * scale_factor for _ in 0:closed_loop_horizon-1], color = :black, linestyle = :dash)

                # 1b. Visualize vertical speed
                axis3 = Axis(fig2[1, 2]; xlabel = "time step", ylabel = "lateral speed [m/s]", xgridvisible = false, ygridvisible = false)
                scatterlines!(axis3, 0:closed_loop_horizon-1, blue_trajectory_vys, label = "Vehicle 1", color = :blue)
                scatterlines!(axis3, 0:closed_loop_horizon-1, red_trajectory_vys, label = "Vehicle 2", color = :red)
                scatterlines!(axis3, 0:closed_loop_horizon-1, green_trajectory_vys, label = "Vehicle 3", color = :green)
                lines!(axis3, 0:closed_loop_horizon-1, [0.2 * scale_factor for _ in 0:closed_loop_horizon-1], color = :black, linestyle = :dash)

                # Visualize distance from goal position
                axis4 = Axis(fig2[2, 1]; xlabel = "time step", ylabel = "Anticipated final distance to goal [m]", xgridvisible = false, ygridvisible = false)
                closed_loop_strategy_end_position1 = Vector{Float64}[]
                closed_loop_strategy_end_position2 = Vector{Float64}[]
                closed_loop_strategy_end_position3 = Vector{Float64}[]
                for i in 1:n_time
                    if method == "GOOP"
                        data = load_object("data/Highway_closed_loop/$(method)_solution/Highway_39_w1_T$(i)_sol.jld2")
                    else
                        data = load_object("data/Highway_closed_loop/$(method)_solution/rfp_39_B2_T$(i)_sol.jld2")
                    end
                    push!(closed_loop_strategy_end_position1, data["strategy1"].xs[end])
                    push!(closed_loop_strategy_end_position2, data["strategy2"].xs[end])
                    push!(closed_loop_strategy_end_position3, data["strategy3"].xs[end])
                end
                scatterlines!(axis4, 0:closed_loop_horizon-1, scale_factor .* [goal_position[1] - closed_loop_strategy_end_position1[k][1]  for k in 1:closed_loop_horizon], label = "Vehicle 1", color = :blue)
                scatterlines!(axis4, 0:closed_loop_horizon-1, scale_factor .* [goal_position[1] - closed_loop_strategy_end_position2[k][1] for k in 1:closed_loop_horizon], label = "Vehicle 2", color = :red)
                scatterlines!(axis4, 0:closed_loop_horizon-1, scale_factor .* [goal_position[1] - closed_loop_strategy_end_position3[k][1] for k in 1:closed_loop_horizon], label = "Vehicle 3", color = :green)

                # Visualize distance between vehicles
                                 # Visualize distance between vehicles using closed-loop data
                axis5 = Axis(fig2[2, 2];
                    xlabel = "time step",
                    ylabel = "inter-vehicle distance [m]",
                    xgridvisible = false, ygridvisible = false,
                    limits = (nothing, (0, 1.5 * scale_factor)),
                )
                
                # Compute pairwise Euclidean distances for each time step
                distances_12 = [sqrt((blue_trajectory_xs[i] - red_trajectory_xs[i])^2 +
                                    (blue_trajectory_ys[i] - red_trajectory_ys[i])^2) for i in 1:closed_loop_horizon]
                distances_13 = [sqrt((blue_trajectory_xs[i] - green_trajectory_xs[i])^2 +
                                    (blue_trajectory_ys[i] - green_trajectory_ys[i])^2) for i in 1:closed_loop_horizon]
                distances_23 = [sqrt((red_trajectory_xs[i] - green_trajectory_xs[i])^2 +
                                    (red_trajectory_ys[i] - green_trajectory_ys[i])^2) for i in 1:closed_loop_horizon]
                
                # Plot the distances with distinct markers and solid lines
                scatterlines!(axis5, 0:closed_loop_horizon-1, distances_12,
                    label = "Vehicle 1 & Vehicle 2",
                    marker = :circle,
                    markersize = 10,
                    linestyle = :solid
                )
                scatterlines!(axis5, 0:closed_loop_horizon-1, distances_13,
                    label = "Vehicle 1 & Vehicle 3",
                    marker = :diamond,
                    markersize = 10,
                    linestyle = :solid
                )
                scatterlines!(axis5, 0:closed_loop_horizon-1, distances_23,
                    label = "Vehicle 2 & Vehicle 3",
                    marker = :utriangle,
                    markersize = 10,
                    linestyle = :solid
                )
                # Plot a dashed threshold line at distance 0.2 m
                threshold = 0.2 * scale_factor
                lines!(axis5, 0:closed_loop_horizon-1, [threshold for _ in 0:closed_loop_horizon-1],
                    color = :black,
                    linestyle = :dash,
                    linewidth = 2,
                    label = "Collision Threshold"
                )
                # Add annotation text to indicate that all distances are safe
                min_distance = minimum(vcat(distances_12, distances_13, distances_23))
                text!(axis5, 2.5, 0.18 * scale_factor, text = "All inter-vehicle distances > safety distance", align = (:center, :top), color = :black, fontsize = 12)
                
                axislegend(axis5)

                fig2[3,1:2] = Legend(fig2, axis2, framevisible = false, orientation = :horizontal)

            end
        end
    end
    
    return fig1, fig2
end

# Specify GOOP or Baseline
method = "Baseline"
# B1: [2500.0, 50.0, 1.0],    # α = 50
# B2: [2500.0, 0.0, 0.0],    # NOTE: Solver failure at T=8 time step for this baseline
fig1, lanes = create_map(n_time = 7)
fig2 = create_data_fig() # visualize speed, distance, etc.
fig1, fig2 = draw_highway_closed_loop!(fig1, lanes, fig2; n_time = 7, method = method)

# Save the figure
display(fig1)
CairoMakie.save("./data/Highway_closed_loop/$(method)_plots/Highway_closed_loop2.pdf", fig1)
display(fig2)
CairoMakie.save("./data/Highway_closed_loop/$(method)_plots/Highway_closed_loop_data2.pdf", fig2)