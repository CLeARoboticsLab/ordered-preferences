module Plot_results_Pareto

using JLD2, CairoMakie, ProgressMeter,  StatsBase

# penalties = [[α^2, α, 1.0] for α = 1:50]

function plot_goop_vs_penalty(;num_samples=2, num_penalty=50)

    colors = 1:num_penalty
    colorrange = (1, num_penalty)
    colormap = :viridis
    baseline_data = Dict[]

    # @showprogress for ii in [39, 97]  # or setdiff(1:num_samples, [23, 24, 25, 26, 27, 28])
    #     filename = "rfp_$(ii)_sol.jld2"
    #     goop_data = load_object("data/relaxably_feasible/GOOP_solution/$filename")
        
    #     # Load baseline data
    #     for jj in 1:num_penalty
    #         baseline_filename = "rfp_$(ii)_baseline$(jj)_sol.jld2"
    #         push!(baseline_data, load_object("data/data_pareto/Baseline_solution/$ii/$baseline_filename"))
    #     end
    
    #     #############################################
    #     # Preprocessing: Compute global min and max #
    #     #############################################
    #     xs_all = Float64[]
    #     ys_all = Float64[]
        
    #     # Collect x and y from each baseline dataset
    #     for jj in 1:num_penalty
    #         xs_temp = baseline_data[jj]["slacks"][1] + baseline_data[jj]["slacks"][3] + baseline_data[jj]["slacks"][5]
    #         ys_temp = baseline_data[jj]["slacks"][2] + baseline_data[jj]["slacks"][4] + baseline_data[jj]["slacks"][6]
    #         append!(xs_all, xs_temp)
    #         append!(ys_all, ys_temp)
    #     end
        
    #     # Collect from GOOP data as well
    #     xs_goop = goop_data["slacks"][1] + goop_data["slacks"][3] + goop_data["slacks"][5]
    #     ys_goop = goop_data["slacks"][2] + goop_data["slacks"][4] + goop_data["slacks"][6]
    #     append!(xs_all, xs_goop)
    #     append!(ys_all, ys_goop)
        
    #     # Get global min and max for normalization
    #     x_min = minimum(xs_all)
    #     x_max = maximum(xs_all)
    #     y_min = minimum(ys_all)
    #     y_max = maximum(ys_all)
        
    #     #############################################
    #     # Plotting: Create normalized scatter plots #
    #     #############################################
    #     # Here we use fixed ticks ([0,0.5,1.0]) since data is now in [0,1]
    #     fig = CairoMakie.Figure(size = (300, 400))
    #     ax1 = CairoMakie.Axis(
    #         fig[2, 1],
    #         xtickformat = "{:.1f}",
    #         xticks = [0.0, 0.5, 1.0],
    #         yticks = [0.0, 0.5, 1.0],
    #         xlabel = L"J^1_3", ylabel = L"J^1_2"
    #     )
        
    #     # Process and plot each baseline dataset
    #     for jj in 1:num_penalty
    #         # Compute original xs and ys for the current baseline dataset
    #         xs = baseline_data[jj]["slacks"][1] + baseline_data[jj]["slacks"][3] + baseline_data[jj]["slacks"][5]
    #         ys = baseline_data[jj]["slacks"][2] + baseline_data[jj]["slacks"][4] + baseline_data[jj]["slacks"][6]
            
    #         # Normalize xs and ys using the global min and max values
    #         xs_norm = (xs .- x_min) ./ (x_max - x_min)
    #         ys_norm = (ys .- y_min) ./ (y_max - y_min)
        
    #         # Plot scatter with or without opacity depending on penalty index
    #         if jj in (10, 30, 50)  # no opacity for some points
    #             CairoMakie.scatter!(ax1, Point2f.(xs_norm, ys_norm), color = colors[jj], colormap = colormap, colorrange = colorrange)
    #         else
    #             CairoMakie.scatter!(ax1, Point2f.(xs_norm, ys_norm), color = colors[jj], colormap = (colormap, 0.5), colorrange = colorrange)
    #         end
    #     end
        
    #     # Normalize and plot GOOP data
    #     xs_norm_goop = (xs_goop .- x_min) ./ (x_max - x_min)
    #     ys_norm_goop = (ys_goop .- y_min) ./ (y_max - y_min)
    #     CairoMakie.scatter!(ax1, Point2f.(xs_norm_goop, ys_norm_goop), color = :black, marker = :star5, markersize = 12)
        
    #     # Add colorbar, then save the figure
    #     CairoMakie.Colorbar(fig[1,1], colormap = colormap, label = L"\alpha", vertical = false, limits = (0, num_penalty))
    #     CairoMakie.save("data/data_pareto/[Pareto] rfp_GOOP_Baseline_$(ii)_2_normalized.pdf", fig, px_per_unit = 8.33)
        
    #     # Reinitialize baseline_data for the next round
    #     baseline_data = Dict[]
    # end

    @showprogress for ii in setdiff(1:num_samples, [23, 24, 25, 26, 27, 28, 32, 56, 60, 67, 69, 71,72, 74, 75, 88, 94])# BASELINE FAILS AT ALPHA 50
        filename = "rfp_$(ii)_sol.jld2"
        goop_data = load_object("data/relaxably_feasible/GOOP_solution/$filename")
        for jj in 1:num_penalty
            baseline_filename = "rfp_$(ii)_baseline$(jj)_sol.jld2"
            push!(baseline_data, load_object("data/data_pareto/Baseline_solution/$baseline_filename"))
        end
        
        # Plot for sum over all player 
        fig = CairoMakie.Figure(size = (300, 400))
        ax1 = CairoMakie.Axis(fig[2, 1], xtickformat = "{:.1f}", xticks = WilkinsonTicks(3), yticks = WilkinsonTicks(3), xlabel=L"J^1_3", ylabel=L"J^1_2")
        for jj in 1:num_penalty
            xs = baseline_data[jj]["slacks"][1]
            ys = baseline_data[jj]["slacks"][2]
            if jj in (10, 30, 50) # no opacity for some points
                CairoMakie.scatter!(ax1, Point2f.(xs,ys), color=colors[jj], colormap=colormap, colorrange=colorrange)
            else
                CairoMakie.scatter!(ax1, Point2f.(xs,ys), color=colors[jj], colormap=(colormap, 0.5), colorrange=colorrange)
            end
        end
        xs = goop_data["slacks"][1]
        ys = goop_data["slacks"][2]
        CairoMakie.scatter!(ax1, Point2f.(xs,ys), color=:black, marker=:star5, markersize = 12) 
        CairoMakie.Colorbar(fig[1,1], colormap=colormap, label=L"\alpha", vertical=false, limits=(0, num_penalty))

        CairoMakie.save("data/data_pareto/[Pareto] rfp_GOOP_Baseline_$(ii)_2_no_sum" * ".pdf", fig, px_per_unit = 8.33)
        # Initialize baseline_data for next round
        baseline_data = Dict[]
    end
end


# Plot
plot_goop_vs_penalty(;num_samples=100, num_penalty=50)



end




