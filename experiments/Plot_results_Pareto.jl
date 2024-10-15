module Plot_results_Pareto

using JLD2, CairoMakie, ProgressMeter

# penalties = [[α^2, α, 1.0] for α = 1:50]

function plot_goop_vs_penalty(;num_samples=2, num_penalty=50)

    colors = 1:num_penalty
    colorrange = (1, num_penalty)
    colormap = :viridis
    baseline_data = Dict[]

    @showprogress for ii in [39, 97]#setdiff(1:num_samples, [23, 24, 25, 26, 27, 28])#
        filename = "rfp_$(ii)_sol.jld2"
        goop_data = load_object("data/relaxably_feasible/GOOP_solution/$filename")
        for jj in 1:num_penalty
            baseline_filename = "rfp_$(ii)_baseline$(jj)_sol.jld2"
            push!(baseline_data, load_object("data_pareto/Baseline_solution/$ii/$baseline_filename"))
        end
        
        # Plot for sum over all player 
        fig = CairoMakie.Figure(size = (300, 400))
        ax1 = CairoMakie.Axis(fig[2, 1], xtickformat = "{:.1f}", xticks = WilkinsonTicks(3), yticks = WilkinsonTicks(3), xlabel=L"\sum_i {J}^i_3", ylabel=L"\sum_i {J}^i_2")
        for jj in 1:num_penalty
            xs = baseline_data[jj]["slacks"][1] + baseline_data[jj]["slacks"][3] + baseline_data[jj]["slacks"][5]
            ys = baseline_data[jj]["slacks"][2] + baseline_data[jj]["slacks"][4] + baseline_data[jj]["slacks"][6]
            if jj in (10, 30, 50) # no opacity for some points
                CairoMakie.scatter!(ax1, Point2f.(xs,ys), color=colors[jj], colormap=colormap, colorrange=colorrange)
            else
                CairoMakie.scatter!(ax1, Point2f.(xs,ys), color=colors[jj], colormap=(colormap, 0.5), colorrange=colorrange)
            end
        end
        xs = goop_data["slacks"][1] + goop_data["slacks"][3] + goop_data["slacks"][5]
        ys = goop_data["slacks"][2] + goop_data["slacks"][4] + goop_data["slacks"][6]
        CairoMakie.scatter!(ax1, Point2f.(xs,ys), color=:black, marker=:star5, markersize = 12) 
        CairoMakie.Colorbar(fig[1,1], colormap=colormap, label=L"\alpha", vertical=false, limits=(0, num_penalty))

        CairoMakie.save("./data_pareto/[Pareto] rfp_GOOP_Baseline_$ii" * ".png", fig, px_per_unit = 8.33)
        Main.@infiltrate
        # Initialize baseline_data for next round
        baseline_data = Dict[]
    end
end


# Plot
plot_goop_vs_penalty(;num_samples=2, num_penalty=50)



end