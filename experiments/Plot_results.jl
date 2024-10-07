module Plot_results

using JLD2, CairoMakie

# 8e-2: [100, 78, 71, 67, 64, 51, 46, 38]
# 1e-1: [100, 78, 71, 67, 64,57, 51, 46, 38, 37, 20, 17]
# 2e-1: [1, 10,13,16,17,20, 38, 39, 42, 46, 51, 55, 57, 64, 65, 67, 71, 75, 78, 96, 100]

function plot_goop_vs_penalty(;num_samples=100, num_penalty=10)

    colors = 1:num_penalty
    colorrange = (1, num_penalty)
    colormap = :viridis
    baseline_label = ["1", "2", "5", "10", "15", "20", "25", "30", "35", "40"]
    baseline_data = Dict[]
    cm_to_pt = 28.346456692913385

    for ii in 1:num_samples
        filename = "rfp_$(ii)_sol.jld2"
        goop_data = load_object("data/relaxably_feasible/GOOP_solution/$filename")
        for jj in 1:num_penalty
            push!(baseline_data, load_object("data/relaxably_feasible/Baseline_solution/$jj/$filename"))
        end
        
        # Plot for player 1
        desired_size = [20, 10] #cm, double column
        fig = CairoMakie.Figure(size = (desired_size[1]*cm_to_pt, desired_size[2]*cm_to_pt))
        ax1 = CairoMakie.Axis(fig[1, 1], xtickformat = "{:.1f}", xticks = WilkinsonTicks(3), yticks = WilkinsonTicks(3), xlabel=L"\textbf{s}^1_3", ylabel=L"\textbf{s}^1_2", title="Player 1")
        for jj in 1:num_penalty
            xs = baseline_data[jj]["slacks"][1]
            ys = baseline_data[jj]["slacks"][2]
            CairoMakie.scatter!(ax1, Point2f.(xs,ys), color=colors[jj], colormap=colormap, colorrange=colorrange, label=baseline_label[jj])
        end
        xs = goop_data["slacks"][1]
        ys = goop_data["slacks"][2]
        CairoMakie.scatter!(ax1, Point2f.(xs,ys), color=:blue)

        # Plot for player 2
        ax2 = CairoMakie.Axis(fig[1, 2], xtickformat = "{:.1f}", ytickformat = "{:.1f}",
            xticks = WilkinsonTicks(2), yticks = WilkinsonTicks(2),
            limits = (-0.1, 0.1, 0.6, 0.8),
            xlabel=L"\textbf{s}^2_3", ylabel=L"\textbf{s}^2_2", title="Player 2")
        for jj in 1:num_penalty
            xs = baseline_data[jj]["slacks"][3]
            ys = baseline_data[jj]["slacks"][4]
            CairoMakie.scatter!(ax2, Point2f.(xs,ys), color=colors[jj], colormap=colormap, colorrange=colorrange, label=baseline_label[jj])
        end
        xs = goop_data["slacks"][3]
        ys = goop_data["slacks"][4]
        CairoMakie.scatter!(ax2, Point2f.(xs,ys), color=:blue)

        # Plot for player 3
        ax3 = CairoMakie.Axis(fig[1, 3], xtickformat = "{:.1f}", ytickformat = "{:.1f}",
            limits = (-0.1, 0.1, 0.3, 0.5),
            xticks = WilkinsonTicks(2), yticks = WilkinsonTicks(2),
            xlabel=L"\textbf{s}^3_3", ylabel=L"\textbf{s}^3_2", title="Player 3")
        for jj in 1:num_penalty
            xs = baseline_data[jj]["slacks"][5]
            ys = baseline_data[jj]["slacks"][6]
            CairoMakie.scatter!(ax3, Point2f.(xs,ys), color=colors[jj], colormap=colormap, colorrange=colorrange, label=baseline_label[jj])
        end
        xs = goop_data["slacks"][5]
        ys = goop_data["slacks"][6]
        CairoMakie.scatter!(ax3, Point2f.(xs,ys), color=:blue)

        fig[2,1:3] = CairoMakie.Legend(fig, ax1, framevisible = false, orientation = :horizontal, nbanks = 1)
        rowgap!(fig.layout, 5)
        CairoMakie.save("./data/relaxably_feasible/result_plots/rfp_GOOP_Baseline_$ii" * ".pdf", fig, pt_per_unit = 1)
        fig

        # Plot level 3 sum of slacks
        desired_size = [13, 9.2] #cm, double column
        fig = CairoMakie.Figure(size = (desired_size[1]*cm_to_pt, desired_size[2]*cm_to_pt))
        xticks = (0:2:num_penalty, vcat(["G"], ["B$i" for i in 2:2:num_penalty]))
        ax1 = CairoMakie.Axis(fig[1, 1], xticks = xticks, yticks = WilkinsonTicks(3), xlabel="Method", title=L"\sum \textbf{s}^i_3")
        CairoMakie.scatter!(ax1, Point2f(0, sum(goop_data["slacks"][1:2:end])), color=:blue, label="GOOP")
        for jj in 1:num_penalty
            CairoMakie.scatter!(ax1, Point2f(jj, sum(baseline_data[jj]["slacks"][1:2:end])), color=colors[jj], colormap=colormap, colorrange=colorrange, label=baseline_label[jj])
        end

        # Plot level 2 sum of slacks
        ax2 = CairoMakie.Axis(fig[1, 2], xticks = xticks, yticks = WilkinsonTicks(3), xlabel="Method", title=L"\sum \textbf{s}^i_2")
        CairoMakie.scatter!(ax2, Point2f(0, sum(goop_data["slacks"][2:2:end])), color=:blue, label="GOOP")
        for jj in 1:num_penalty
            CairoMakie.scatter!(ax2, Point2f(jj, sum(baseline_data[jj]["slacks"][2:2:end])), color=colors[jj], colormap=colormap, colorrange=colorrange, label=baseline_label[jj])
        end

        CairoMakie.save("./data/relaxably_feasible/result_plots/rfp_GOOP_Baseline_Sum_Slacks_$ii" * ".pdf", fig, pt_per_unit = 1)
        fig

        # Main.@infiltrate
        # Plot Monte Carlo result



    end
end


# Plot
plot_goop_vs_penalty(;num_samples=100, num_penalty=10)



end