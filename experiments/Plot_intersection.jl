using CairoMakie
using Makie.GeometryBasics  # Required for defining polygons

function draw_intersection_scaled()
    lane_width = 2  # Width of a lane 
    map_end = 7  # Length of the road 
    offset = 0.2
    fig = Figure(size = (700, 700))
    ax = Axis(fig[1, 1], aspect = 1, xgridvisible = false, ygridvisible = false, backgroundcolor = :lightgreen)
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


   
    fig
end
fig = draw_intersection_scaled()

save("GOOP_intersection.pdf", fig)