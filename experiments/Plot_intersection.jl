using CairoMakie
using Makie.GeometryBasics  # Required for defining polygons
using JLD2
using TrajectoryGamesBase: OpenLoopStrategy

function create_map()
    lane_width = 2  # Width of a lane 
    map_end = 7 # Length of the road 
    offset = 0.2
    fig = Figure(size = (500, 500))
    ax = Axis(fig[1:6, 1:6], aspect = 1, xgridvisible = false, ygridvisible = false)
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
 
     arrows!(xs, ys, us, vs; arrowsize = 15, lengthscale = 0.5, arrowcolor = :white, linecolor = :white, linewidth = 3)
    return fig, ax
end
function create_fig2()
    # Visualize vertical speed, horizontal speed, ...
    fig = Figure(size = (1000, 700))
    return fig
end

function draw_intersection_open_loop!(fig1, ax1, fig2)
    scale_factor = 10 * 2/3
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
    blue_trajectory_xs = [data[1] for data in goop_strategy1_xs] .* scale_factor
    blue_trajectory_ys = [data[2] for data in goop_strategy1_xs] .* scale_factor
    blue_trajectory_vxs = [data[3] for data in goop_strategy1_xs] .* scale_factor
    blue_trajectory_vys = [data[4] for data in goop_strategy1_xs] .* scale_factor

    for i in 1:length(blue_trajectory_xs)-1
        scatterlines!(
            ax1,
            [blue_trajectory_xs[i], blue_trajectory_xs[i+1]] ./ scale_factor,
            [blue_trajectory_ys[i], blue_trajectory_ys[i+1]] ./ scale_factor,
            color = blue_trajectory_vxs[i],
            colormap = :blues,
            colorrange = (0, 20),
            linewidth = 2
        )
    end
    Colorbar(fig1[3, 2:5], limits = (0, 20), flipaxis = false, label = "Vehicle 1's speed [m/s]", colormap = :blues, vertical = false)

    # Plot red trajectory: explicitly show distance-to-goal relaxation
    red_trajectory_xs = [data[1] for data in goop_strategy2_xs] .* scale_factor
    red_trajectory_ys = [data[2] for data in goop_strategy2_xs] .* scale_factor
    red_trajectory_vxs = [data[3] for data in goop_strategy2_xs] .* scale_factor
    red_trajectory_vys = [data[4] for data in goop_strategy2_xs] .* scale_factor

    for i in 1:length(red_trajectory_xs)-1
        scatterlines!(
            ax1,
            [red_trajectory_xs[i], red_trajectory_xs[i+1]] ./ scale_factor,
            [red_trajectory_ys[i], red_trajectory_ys[i+1]] ./ scale_factor,
            color = :red,
            linewidth = 2
        )
    end
    lines!(
        ax1,
        [red_trajectory_xs[end] / scale_factor, goal_position2[1]],
        [red_trajectory_ys[end] / scale_factor, goal_position2[2]],
        color = :purple,
        linewidth = 2,
        linestyle = :dash
    )

    # Visualize initial states 
    scatter!(
        ax1,
        [Point2f(initial_state1), Point2f(initial_state2)],
        markersize = 20,
        color = [:blue, :red]
    )

    # Visualize goal positions
    scatter!(
        ax1,
        Point2f(goal_position1),
        markersize = 20,
        marker = :star5,
        color = :blue,
    )
    scatter!(
        ax1,
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
    axis2 = Axis(fig2[1, 1]; xlabel = "time step", ylabel = "speed [m/s]", title = "Horizontal Speed")
    for i in 1:length(blue_trajectory_xs)-1
        scatterlines!(
            axis2,
            [i-1, i],
            [blue_trajectory_vxs[i], blue_trajectory_vxs[i+1]],
            color = blue_trajectory_vxs[i],
            colormap = :blues,
            colorrange = (0, 30),
            linewidth = 2,
            label = "Robot 1"
        )
    end
    scatterlines!(axis2, 0:planning_horizon-1, red_trajectory_vxs, label = "Robot 2", color = :red)
    lines!(axis2, 0:planning_horizon-1, [15 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)

    # Visualize vertical speed
    axis3 = Axis(fig2[1, 2]; xlabel = "time step", ylabel = "speed [m/s]", title = "Vertical Speed")
    scatterlines!(axis3, 0:planning_horizon-1, blue_trajectory_vys, label = "Vehicle 1", color = :blue)
    scatterlines!(axis3, 0:planning_horizon-1, red_trajectory_vys, label = "Vehicle 2", color = :red)
    lines!(axis3, 0:planning_horizon-1, [15 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)

    # Visualize distance bw vehicles , limits = (nothing, (collision_avoidance-0.05, 0.4)) 
    axis4 = Axis(fig2[2, 1]; xlabel = "time step", ylabel = "distance [m]", title = "Distance bw robots")
    scatterlines!(axis4, 0:planning_horizon-1, openloop_distance1[T], label = "B/w Robot 1 & Robot 2", color = :black, marker = :star5, markersize = 20)
    lines!(axis4, 0:planning_horizon-1, [1.0 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)

    # Visualize distance from center yellow line
    axis5 = Axis(fig2[2, 2]; xlabel = "time step",  ylabel = "displacement [m]", title = "Relative position from yellow line")
    scatterlines!(axis5, 0:planning_horizon-1, blue_trajectory_ys, label = "Vehicle 1", color = :blue)
    scatterlines!(axis5, 0:planning_horizon-1, -red_trajectory_xs, label = "Vehicle 2", color = :red)
    lines!(axis5, 0:planning_horizon-1, [0.0 for _ in 0:planning_horizon-1], color = :black, linestyle = :dash)
    
    fig2[3,1:2] = Legend(fig2, axis5, framevisible = false, orientation = :horizontal)

    fig1, fig2
end

function draw_intersection_closed_loop!(fig1, ax1, fig2, fig3, ax3, fig4, ax4, fig5, ax5, fig6, ax6)

    scale_factor = 10 * 2/3
    cmap_max = 20
    # Form closed_loop trajectory data
    n_step = 15
    closed_loop_strategy1 = Vector{Float64}[]
    closed_loop_strategy2 = Vector{Float64}[]

    for i in 1:n_step
        goop_data = load_object("data/Intersection_closed_loop/GOOP_solution/intersection_$(i).jld2")
        push!(closed_loop_strategy1, first(goop_data["strategy1"].xs))
        push!(closed_loop_strategy2, first(goop_data["strategy2"].xs))
    end

    # Load problem data 
    problem_data = load_object("data/Intersection/problem/problem_data_1.jld2") # from open loop data
    initial_state1 = problem_data["initial_state1"]
    initial_state2 = problem_data["initial_state2"]
    goal_position1 = problem_data["goal_position1"]
    goal_position2 = problem_data["goal_position2"]
    

    # Plot blue trajectory: color gradient to show speed violations
    blue_trajectory_xs = [data[1] for data in closed_loop_strategy1] .* scale_factor
    blue_trajectory_ys = [data[2] for data in closed_loop_strategy1] .* scale_factor
    blue_trajectory_vxs = [data[3] for data in closed_loop_strategy1] .* scale_factor
    blue_trajectory_vys = [data[4] for data in closed_loop_strategy1] .* scale_factor

    function plot_blue_trajectory(ax; T = length(blue_trajectory_xs)-1)
        for i in 1:T
            scatterlines!(
                ax,
                [blue_trajectory_xs[i], blue_trajectory_xs[i+1]] ./ scale_factor,
                [blue_trajectory_ys[i], blue_trajectory_ys[i+1]] ./ scale_factor,
                color = blue_trajectory_vxs[i],
                colormap = (:blues, 0.6), # transparency
                colorrange = (0, 20),
                linewidth = 2,
                linestyle = (:dash, :dense),
                marker = :diamond
            )
        end
    end
    plot_blue_trajectory(ax1; T = 4)
    Colorbar(fig1[6, 2:5], limits = (0, 20), flipaxis = false, label = "Vehicle 1's speed [m/s]", colormap = :blues, vertical = false)

    # Plot red trajectory: explicitly show distance-to-goal relaxation
    red_trajectory_xs = [data[1] for data in closed_loop_strategy2] .* scale_factor
    red_trajectory_ys = [data[2] for data in closed_loop_strategy2] .* scale_factor
    red_trajectory_vxs = [data[3] for data in closed_loop_strategy2] .* scale_factor
    red_trajectory_vys = [data[4] for data in closed_loop_strategy2] .* scale_factor

    function plot_red_trajectory(ax; T = length(red_trajectory_xs)-1)
        for i in 1:T
            scatterlines!(
                ax,
                [red_trajectory_xs[i], red_trajectory_xs[i+1]] ./ scale_factor,
                [red_trajectory_ys[i], red_trajectory_ys[i+1]] ./ scale_factor,
                color = red_trajectory_vys[i],
                colormap = (:reds, 0.6), # transparency
                colorrange = (0, 20),
                linewidth = 2,
                linestyle = (:dash, :dense),
                marker = :diamond
            )
        end
    end
    plot_red_trajectory(ax1; T = 4)
    Colorbar(fig1[1, 2:5], limits = (0, 20), flipaxis = true, label = "Vehicle 2's speed [m/s]", colormap = :reds, vertical = false)

    # # Visualize initial states 
    # scatter!(
    #     ax1,
    #     Point2f(initial_state1),
    #     markersize = 20,
    #     color = (:blue, 0.5)
    # )
    # scatter!(
    #     ax1,
    #     Point2f(initial_state2),
    #     markersize = 20,
    #     color = (:red,0.5)
    # )

    # Plot interim open-loop trajectories
    T = 4
    fig1, ax1 = plot_trajectories!(fig1, ax1, scale_factor; T)
    
    # Visualize goal positions
    function plot_goal_positions!(ax1, goal_position1, goal_position2)
        scatter!(
            ax1,
            Point2f(goal_position1),
            markersize = 20,
            marker = :star5,
            color = :blue,
        )
        scatter!(
            ax1,
            Point2f(goal_position2),
            markersize = 20,
            marker = :star5,
            color = :red,
        )
    end
    plot_goal_positions!(ax1, goal_position1, goal_position2)

    # Store speed and distance data for Intersection (closed_loop)
    closed_loop_horizon = length(closed_loop_strategy1)
    closed_loop_distance1 = [sqrt(sum((closed_loop_strategy1[k][1:2] - closed_loop_strategy2[k][1:2]) .^ 2)) for k in 1:closed_loop_horizon]

    # 1a. Visualize horizontal speed
    axis2 = Axis(fig2[1, 1]; xlabel = "time step", ylabel = "speed [m/s]", xgridvisible = false, ygridvisible = false, xlabelsize = 28, ylabelsize = 26, xticklabelsize = 28, yticklabelsize = 28)
    for i in 1:length(blue_trajectory_xs)-1
        scatterlines!(
            axis2,
            [i-1, i],
            [blue_trajectory_vxs[i], blue_trajectory_vxs[i+1]],
            color = blue_trajectory_vxs[i],
            colormap = :blues,
            colorrange = (0, 20),
            linewidth = 3.5,
            markersize = 15,
            label = "Robot 1",
        )
    end
    # scatterlines!(axis2, 0:closed_loop_horizon-1, red_trajectory_vxs, label = "Robot 2", color = :red)
    lines!(axis2, 0:closed_loop_horizon-1, [1.5*scale_factor for _ in 0:closed_loop_horizon-1], color = :black, linestyle = :dash)

    # 1b. Visualize vertical speed
    for i in 1:length(red_trajectory_xs)-1
        scatterlines!(
            axis2,
            [i-1, i],
            [red_trajectory_vys[i], red_trajectory_vys[i+1]],
            color = abs(red_trajectory_vys[i]),
            colormap = :reds,
            colorrange = (0, 20),
            linewidth = 3.5,
            markersize = 15,
            label = "Robot 2"
        )
    end
    # scatterlines!(ax3, 0:closed_loop_horizon-1, blue_trajectory_vys, label = "Vehicle 1", color = :blue)
    lines!(axis2, 0:closed_loop_horizon-1, [1.5*scale_factor for _ in 0:closed_loop_horizon-1], color = :black, linestyle = :dash)

    # Visualize distance from goal position
    axis3 = Axis(fig2[1, 2]; xlabel = "time step", ylabel = "goal-reaching error [m]", xgridvisible = false, ygridvisible = false, xlabelsize = 28, ylabelsize = 26, xticklabelsize = 28, yticklabelsize = 28)
    closed_loop_strategy_end_position1 = Vector{Float64}[]
    closed_loop_strategy_end_position2 = Vector{Float64}[]
    for i in 1:n_step
        goop_data = load_object("data/Intersection_closed_loop/GOOP_solution/intersection_$(i).jld2")
        push!(closed_loop_strategy_end_position1, goop_data["strategy1"].xs[end])
        push!(closed_loop_strategy_end_position2, goop_data["strategy2"].xs[end])
    end
    scatterlines!(axis3, 0:closed_loop_horizon-1, scale_factor .* [sqrt(sum((closed_loop_strategy_end_position1[k][1:2] - goal_position1) .^ 2)) for k in 1:closed_loop_horizon], label = "Robot 2", color = :blue, linewidth = 3.5, markersize = 15)
    scatterlines!(axis3, 0:closed_loop_horizon-1, scale_factor .* [sqrt(sum((closed_loop_strategy_end_position2[k][1:2] - goal_position2) .^ 2)) for k in 1:closed_loop_horizon], label = "Robot 2", color = :red, linewidth = 3.5, markersize = 15)

    # Visualize distance bw vehicles , limits = (nothing, (collision_avoidance-0.05, 0.4)) 
    axis4 = Axis(fig2[2, 1]; xlabel = "time step", ylabel = "distance [m]", title = "Distance bw robots", xgridvisible = false, ygridvisible = false)
    scatterlines!(axis4, 0:closed_loop_horizon-1, scale_factor .* closed_loop_distance1, label = "B/w Robot 1 & Robot 2", color = :black, linewidth = 3.5, markersize = 15)
    lines!(axis4, 0:closed_loop_horizon-1, scale_factor .* [1.3 for _ in 0:closed_loop_horizon-1], color = :black, linestyle = :dash)

    # Visualize distance from center yellow line
    axis5 = Axis(fig2[2, 2]; xlabel = "time step",  ylabel = "displacement [m]", title = "Relative position from yellow line", xgridvisible = false, ygridvisible = false)
    scatterlines!(axis5, 0:closed_loop_horizon-1, blue_trajectory_ys, label = "Vehicle 1", color = :blue, linewidth = 3.5, markersize = 15)
    scatterlines!(axis5, 0:closed_loop_horizon-1, -red_trajectory_xs, label = "Vehicle 2", color = :red, linewidth = 3.5, markersize = 15)
    lines!(axis5, 0:closed_loop_horizon-1, scale_factor .* [0.0 for _ in 0:closed_loop_horizon-1], color = :black, linestyle = :dash)
    
    fig2[3,1:2] = Legend(fig2, axis5, framevisible = false, orientation = :horizontal)

    # Visualize trajectories as time progresses
    # 1. T = 1 (open loop)
    fig3, ax3 = plot_trajectories!(fig3, ax3, scale_factor; T=1)
    plot_goal_positions!(ax3, goal_position1, goal_position2)
    # 2. T = 3
    plot_blue_trajectory(ax4; T=3) # past trajectory
    plot_red_trajectory(ax4; T=3)
    fig4, ax4 = plot_trajectories!(fig4, ax4, scale_factor; T=3) # open loop at T=3
    plot_goal_positions!(ax4, goal_position1, goal_position2)
    # 3. T = 4
    plot_blue_trajectory(ax5; T=5)
    plot_red_trajectory(ax5; T=5)
    fig5, ax5 = plot_trajectories!(fig5, ax5, scale_factor; T=4)
    plot_goal_positions!(ax5, goal_position1, goal_position2)
    # 4. T = 6
    plot_blue_trajectory(ax6; T=6)
    plot_red_trajectory(ax6; T=6)
    fig6, ax6 = plot_trajectories!(fig6, ax6, scale_factor; T=6)
    plot_goal_positions!(ax6, goal_position1, goal_position2)

    fig1, fig2, fig3, fig4, fig5, fig6
end

function plot_trajectories!(fig1, ax1, scale_factor; T = 3)

        # Load problem data at time step T = 3,4
        problem_data_t = load_object("data/Intersection_closed_loop/GOOP_solution/intersection_$(T).jld2") # from closed loop data
        goop_strategy1_xs = problem_data_t["strategy1"].xs 
        goop_strategy2_xs = problem_data_t["strategy2"].xs 
    
        # Plot blue trajectory: color gradient to show speed violations
        blue_trajectory_xs = [data[1] for data in goop_strategy1_xs] .* scale_factor
        blue_trajectory_ys = [data[2] for data in goop_strategy1_xs] .* scale_factor
        blue_trajectory_vxs = [data[3] for data in goop_strategy1_xs] .* scale_factor
        # blue_trajectory_vys = [data[4] for data in goop_strategy1_xs] .* scale_factor

        for i in 1:length(blue_trajectory_xs)-1
            scatterlines!(
                ax1,
                [blue_trajectory_xs[i], blue_trajectory_xs[i+1]] ./ scale_factor,
                [blue_trajectory_ys[i], blue_trajectory_ys[i+1]] ./ scale_factor,
                color = blue_trajectory_vxs[i],
                colormap = :blues,
                colorrange = (0, 20),
                linewidth = 1.5,
            )
        end
        # Plot red trajectory: explicitly show distance-to-goal relaxation
        red_trajectory_xs = [data[1] for data in goop_strategy2_xs] .* scale_factor
        red_trajectory_ys = [data[2] for data in goop_strategy2_xs] .* scale_factor
        # red_trajectory_vxs = [data[3] for data in goop_strategy2_xs] .* scale_factor
        red_trajectory_vys = [data[4] for data in goop_strategy2_xs] .* scale_factor

        for i in 1:length(red_trajectory_xs)-1
            scatterlines!(
                ax1,
                [red_trajectory_xs[i], red_trajectory_xs[i+1]] ./ scale_factor,
                [red_trajectory_ys[i], red_trajectory_ys[i+1]] ./ scale_factor,
                color = red_trajectory_vys[i],
                colormap = :reds,
                colorrange = (0, 20),
                linewidth = 1.5,
            )
        end

        # Visualize initial states
        initial_state1 = first(goop_strategy1_xs)[1:2]
        initial_state2 = first(goop_strategy2_xs)[1:2]
        scatter!(
            ax1,
            [Point2f(initial_state1), Point2f(initial_state2)],
            markersize = 15,
            color = [:blue, :red]
        )

    return fig1, ax1
end

fig1, ax1 = create_map() # visualize trajectory
fig2 = create_fig2() # visualize speed, distance, etc.
fig3, ax3 = create_map() # visualize trajectory as time progresses
fig4, ax4 = create_map() 
fig5, ax5 = create_map()
fig6, ax6 = create_map() 

closed_loop = true
if !closed_loop
    fig1, fig2 = draw_intersection_open_loop!(fig1, ax1, fig2)
    save("data/Intersection/GOOP_intersection.pdf", fig1)
    save("data/Intersection/GOOP_intersection_data.pdf", fig2)
else
    fig1, fig2, fig3, fig4, fig5, fig6 = draw_intersection_closed_loop!(fig1, ax1, fig2, fig3, ax3, fig4, ax4, fig5, ax5, fig6, ax6)
    save("data/Intersection_closed_loop/GOOP_intersection_closed_loop.pdf", fig1)
    save("data/Intersection_closed_loop/GOOP_intersection_closed_loop_data.pdf", fig2)
    save("data/Intersection_closed_loop/GOOP_intersection_closed_loop_over_timeT1.pdf", fig3)
    save("data/Intersection_closed_loop/GOOP_intersection_closed_loop_over_timeT3.pdf", fig4)
    save("data/Intersection_closed_loop/GOOP_intersection_closed_loop_over_timeT4.pdf", fig5)
    save("data/Intersection_closed_loop/GOOP_intersection_closed_loop_over_timeT6.pdf", fig6)


end