module Plot_results

using JLD2, CairoMakie


function plot_goop_vs_penalty(;num_samples=100, num_penalty=7)

    colors = [:yellow, :purple, :orange, :brown, :black, :pink, :cyan]
    baseline_label = ["1", "2", "4", "10", "20", "40", "100"]
    
    baseline_data = Dict[]

    for ii in [1]#2:num_samples
        filename = "rfp_$(ii)_sol.jld2"
        goop_data = load_object("data/relaxably_feasible/GOOP_solution/$filename")
        for jj in 1:num_penalty
            push!(baseline_data, load_object("data/relaxably_feasible/Baseline_solution/$jj/$filename"))
        end
        
        # Plot for player 1
        fig = CairoMakie.Figure()
        ax1 = CairoMakie.Axis(fig[1, 1], xlabel=L"\sum s^1_3", ylabel=L"\sum s^1_2", title="Player 1")
        xs = goop_data["slacks"][1]
        ys = goop_data["slacks"][2]
        CairoMakie.scatter!(ax1, Point2f.(xs,ys), color=:blue, label="Player 1")

        for jj in 1:num_penalty
            xs = baseline_data[jj]["slacks"][1]
            ys = baseline_data[jj]["slacks"][2]
            CairoMakie.scatter!(ax1, Point2f.(xs,ys), color=colors[jj], label=baseline_label[jj])
        end
        # CairoMakie.save("./data/relaxably_feasible/result_plots/rfp_GOOP_Baseline1_$ii" * ".png", fig)
        # fig

        # Plot for player 2
        ax2 = CairoMakie.Axis(fig[1, 2], xlabel=L"\sum s^2_3", ylabel=L"\sum s^2_2", title="Player 2")
        xs = goop_data["slacks"][3]
        ys = goop_data["slacks"][4]
        CairoMakie.scatter!(ax2, Point2f.(xs,ys), color=:blue, label="Player 2")

        for jj in 1:num_penalty
            xs = baseline_data[jj]["slacks"][3]
            ys = baseline_data[jj]["slacks"][4]
            CairoMakie.scatter!(ax2, Point2f.(xs,ys), color=colors[jj], label=baseline_label[jj])
        end

        # Plot for player 3
        ax3 = CairoMakie.Axis(fig[1, 3], xlabel=L"\sum s^3_3", ylabel=L"\sum s^3_2", title="Player 3")
        xs = goop_data["slacks"][5]
        ys = goop_data["slacks"][6]
        CairoMakie.scatter!(ax3, Point2f.(xs,ys), color=:blue, label="Player 3")

        for jj in 1:num_penalty
            xs = baseline_data[jj]["slacks"][5]
            ys = baseline_data[jj]["slacks"][6]
            CairoMakie.scatter!(ax3, Point2f.(xs,ys), color=colors[jj], label=baseline_label[jj])
        end

        fig[2,1:3] = CairoMakie.Legend(fig, ax1, framevisible = false, orientation = :horizontal)
        CairoMakie.save("./data/relaxably_feasible/result_plots/rfp_GOOP_Baseline1_$ii" * ".png", fig)
        fig

        # Plot level 3 sum of slacks
        fig = CairoMakie.Figure()
        ax1 = CairoMakie.Axis(fig[1, 1], xticks = 0:num_penalty, xlabel="Method", ylabel="Sum of slacks", title="Comparing sum of slacks at Level = 3")
        CairoMakie.scatter!(ax1, Point2f(0, sum(goop_data["slacks"][1:2:end])), color=:blue, label="GOOP")
        for jj in 1:num_penalty
            CairoMakie.scatter!(ax1, Point2f(jj, sum(baseline_data[jj]["slacks"][1:2:end])), color=colors[jj], label=baseline_label[jj])
        end

        # Plot level 2 sum of slacks
        ax2 = CairoMakie.Axis(fig[1, 2], xticks = 0:num_penalty, xlabel="Method", ylabel="Sum of slacks", title="Comparing sum of slacks at Level = 2")
        CairoMakie.scatter!(ax2, Point2f(0, sum(goop_data["slacks"][2:2:end])), color=:blue, label="GOOP")
        for jj in 1:num_penalty
            CairoMakie.scatter!(ax2, Point2f(jj, sum(baseline_data[jj]["slacks"][2:2:end])), color=colors[jj], label=baseline_label[jj])
        end

        fig[2,1:2] = CairoMakie.Legend(fig, ax1, framevisible = false, orientation = :horizontal)
        CairoMakie.save("./data/relaxably_feasible/result_plots/rfp_GOOP_Baseline_Sum_Slacks_$ii" * ".png", fig)
        fig
    end
end


# Plot 1
plot_goop_vs_penalty(;num_samples=4, num_penalty=7)

# Plot 2


end