using CairoMakie
using Makie.GeometryBasics  # Required for defining polygons
using JLD2
using TrajectoryGamesBase: OpenLoopStrategy

function draw_intersection_scaled()
    lane_width = 2  # Width of a lane 
    map_end = 7 # Length of the road 
    offset = 0.2
    fig1 = Figure(size = (500, 500))
    ax = Axis(fig1[1:3, 1:6], aspect = 1, xgridvisible = false, ygridvisible = false, backgroundcolor = :lightgreen)
    hidedecorations!(ax)
    hidespines!(ax)

    vertical_road_background = Polygon(
        Point2f[(-lane_width-offset, -map_end), (lane_width+offset, -map_end), (lane_width+offset, map_end), (-lane_width-offset, map_end)]
    )
    poly!(vertical_road_background, color = :white)
    lines!(ax, [-lane_width-offset, -lane_width-offset], [-map_end, -lane_width], color = :black, linewidth = 1)
    lines!(ax, [-lane_width-offset, -lane_width-offset], [map_end, lane_width], color = :black, linewidth = 1)
    lines!(ax, [lane_width+offset, lane_width+offset], [-map_end, -lane_width], color = :black, linewidth = 1)
    lines!(ax, [lane_width+offset, lane_width+offset], [map_end, lane_width], color = :black, linewidth = 1)

    horizontal_road_background = Polygon(
        Point2f[(-map_end, -lane_width-offset), (map_end, -lane_width-offset), (map_end, lane_width+offset), (-map_end, lane_width+offset)]
    )
    poly!(horizontal_road_background, color = :white)
    lines!(ax, [-lane_width-offset, -map_end], [-lane_width-offset, -lane_width-offset], color = :black, linewidth = 1)
    lines!(ax, [-lane_width-offset, -map_end], [lane_width+offset, lane_width+offset], color = :black, linewidth = 1)
    lines!(ax, [lane_width+offset, map_end], [lane_width+offset, lane_width+offset], color = :black, linewidth = 1)
    lines!(ax, [lane_width+offset, map_end], [-lane_width-offset, -lane_width-offset], color = :black, linewidth = 1)

    vertical_road = Polygon(
        Point2f[(-lane_width, -map_end), (lane_width, -map_end), (lane_width, map_end), (-lane_width, map_end)]
    )
    poly!(vertical_road, color = :gray)
    horizontal_road = Polygon(
        Point2f[(-map_end, -lane_width), (map_end, -lane_width), (map_end, lane_width), (-map_end, lane_width)]
    )
    poly!(horizontal_road, color = :gray)

    # Lane markings (dashed center lines)
    lines!(ax, [-lane_width, -map_end], [0, 0], color = :yellow, linewidth = 2)
    lines!(ax, [-lane_width, -map_end], [0, 0], color = :yellow, linewidth = 2)
    lines!(ax, [lane_width, map_end], [0, 0], color = :yellow, linewidth = 2)
    lines!(ax, [lane_width, map_end], [0, 0], color = :yellow, linewidth = 2)
    lines!(ax, [0, 0], [-lane_width, -map_end], color = :yellow, linewidth = 2)
    lines!(ax, [0, 0], [-lane_width, -map_end], color = :yellow, linewidth = 2)
    lines!(ax, [0, 0], [lane_width, map_end], color = :yellow, linewidth = 2)
    lines!(ax, [0, 0], [lane_width, map_end], color = :yellow, linewidth = 2)

     # Add directional arrows using arrows!
     xs = [-3, 3, 1, -1]  # Starting x-coordinates for arrows
     ys = [-1, 1, -3, 3]  # Starting y-coordinates for arrows
     us = [1, -1, 0, 0]      # Arrow x-directions
     vs = [0, 0, 1, -1]      # Arrow y-directions
 
     arrows!(xs, ys, us, vs; arrowsize = 15, lengthscale = 0.5,
         arrowcolor = :white, linecolor = :white, linewidth = 3)

    # Visuliaze trajectories
    goop_data = load_object("data/Intersection/GOOP_solution/intersection_1.jld2")
    goop_strategy1_xs = goop_data["strategy1"].xs 
    goop_strategy2_xs = goop_data["strategy2"].xs 

    # Load problem data 
    problem_data = load_object("data/Intersection/problem/problem_data_1.jld2")
    initial_state1 = problem_data["initial_state1"]
    initial_state2 = problem_data["initial_state2"]
    goal_position1 = problem_data["goal_position1"]
    goal_position2 = problem_data["goal_position2"]

    # Plot blue trajectory: color gradient to show speed violations
    blue_trajectory_xs = [data[1] for data in goop_strategy1_xs] .* 10
    blue_trajectory_ys = [data[2] for data in goop_strategy1_xs] .* 10
    blue_trajectory_vxs = [data[3] for data in goop_strategy1_xs] .* 10
    blue_trajectory_vys = [data[4] for data in goop_strategy1_xs] .* 10

    for i in 1:length(blue_trajectory_xs)-1
        scatterlines!(
            ax,
            [blue_trajectory_xs[i], blue_trajectory_xs[i+1]] ./ 10,
            [blue_trajectory_ys[i], blue_trajectory_ys[i+1]] ./ 10,
            color = blue_trajectory_vxs[i],
            colormap = :blues,
            colorrange = (0, 25),
            linewidth = 2
        )
    end
    Colorbar(fig1[3, 2:5], limits = (0, 25), flipaxis = false, label = "Player 1's speed [m/s]", colormap = :blues, vertical = false)

    # Plot red trajectory: explicitly show distance-to-goal relaxation
    red_trajectory_xs = [data[1] for data in goop_strategy2_xs] .* 10
    red_trajectory_ys = [data[2] for data in goop_strategy2_xs] .* 10
    red_trajectory_vxs = [data[3] for data in goop_strategy2_xs] .* 10
    red_trajectory_vys = [data[4] for data in goop_strategy2_xs] .* 10

    for i in 1:length(red_trajectory_xs)-1
        scatterlines!(
            ax,
            [red_trajectory_xs[i], red_trajectory_xs[i+1]] ./ 10,
            [red_trajectory_ys[i], red_trajectory_ys[i+1]] ./ 10,
            color = :red,
            linewidth = 2
        )
    end
    lines!(
        ax,
        [red_trajectory_xs[end] / 10, goal_position2[1]],
        [red_trajectory_ys[end] / 10, goal_position2[2]],
        color = :purple,
        linewidth = 2,
        linestyle = :dash
    )

    # Visualize initial states 
    scatter!(
        ax,
        [Point2f(initial_state1), Point2f(initial_state2)],
        markersize = 20,
        color = [:blue, :red]
    )

    # Visualize goal positions
    scatter!(
        ax,
        Point2f(goal_position1),
        markersize = 20,
        marker = :star5,
        color = :blue,
    )
    scatter!(
        ax,
        Point2f(goal_position2),
        markersize = 20,
        marker = :star5,
        color = :red,
    )

    # Store speed data for Intersection
    planning_horizon = length(goop_strategy1_xs)
    openloop_distance1 = Vector{Float64}[]

    # Store openloop distance data
    push!(openloop_distance1, [sqrt(sum((goop_strategy1_xs[k][1:2] - goop_strategy2_xs[k][1:2]) .^ 2)) for k in 1:planning_horizon])

    # Visualize horizontal speed
    T = 1
    fig = Figure(size = (700, 700))
    ax2 = Axis(fig[1, 1]; xlabel = "time step", ylabel = "speed [m/s]", title = "Horizontal Speed")
    for i in 1:length(blue_trajectory_xs)-1
        scatterlines!(
            ax2,
            [i-1, i],
            [blue_trajectory_vxs[i], blue_trajectory_vxs[i+1]],
            color = blue_trajectory_vxs[i],
            colormap = :blues,
            colorrange = (0, 30),
            linewidth = 2,
            label = "Robot 1"
        )
    end
    scatterlines!(ax2, 0:planning_horizon-1, red_trajectory_vxs, label = "Robot 2", color = :red)
    lines!(ax2, 0:planning_horizon-1, [15 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)

    # Visualize vertical speed
    ax3 = Axis(fig[1, 2]; xlabel = "time step", ylabel = "speed [m/s]", title = "Vertical Speed")
    scatterlines!(ax3, 0:planning_horizon-1, blue_trajectory_vys, label = "Vehicle 1", color = :blue)
    scatterlines!(ax3, 0:planning_horizon-1, red_trajectory_vys, label = "Vehicle 2", color = :red)
    lines!(ax3, 0:planning_horizon-1, [15 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)

    # Visualize distance bw vehicles , limits = (nothing, (collision_avoidance-0.05, 0.4)) 
    ax4 = Axis(fig[2, 1]; xlabel = "time step", ylabel = "distance [m]", title = "Distance bw robots")
    scatterlines!(ax4, 0:planning_horizon-1, openloop_distance1[T], label = "B/w Robot 1 & Robot 2", color = :black, marker = :star5, markersize = 20)
    lines!(ax4, 0:planning_horizon-1, [1.0 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)

    # Visualize distance from center yellow line
    ax5 = Axis(fig[2, 2]; xlabel = "time step",  ylabel = "displacement [m]", title = "Relative position from yellow line")
    scatterlines!(ax5, 0:planning_horizon-1, blue_trajectory_ys, label = "Vehicle 1", color = :blue)
    scatterlines!(ax5, 0:planning_horizon-1, -red_trajectory_xs, label = "Vehicle 2", color = :red)
    lines!(ax5, 0:planning_horizon-1, [0.0 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)
    
    fig[3,1:2] = Legend(fig, ax5, framevisible = false, orientation = :horizontal)

    fig1, fig
end
fig1, fig = draw_intersection_scaled()

save("data/Intersection/GOOP_intersection.pdf", fig1)
save("data/Intersection/GOOP_intersection_data.pdf", fig)