module Plot_results

using JLD2, CairoMakie, ProgressMeter

# 8e-2: [100, 78, 71, 67, 64, 51, 46, 38]
# 1e-1: [100, 78, 71, 67, 64,57, 51, 46, 38, 37, 20, 17]
# 2e-1: [1, 10,13,16,17,20, 38, 39, 42, 46, 51, 55, 57, 64, 65, 67, 71, 75, 78, 96, 100]

function plot_goop_vs_penalty(;num_samples=100, num_penalty=10)

    colors = 1:num_penalty
    colorrange = (1, num_penalty)
    colormap = :viridis
    baseline_label = ["1", "2", "5", "10", "15", "20", "25", "30", "35", "40"]
    baseline_data = Dict[]

    Δ_slacks_3 = Vector[]
    Δ_slacks_2 = Vector[]

    goop_slack = Vector[]
    baseline_slack = Vector[]

    cm_to_pt = 28.346456692913385

    @showprogress for ii in setdiff(1:num_samples, [17, 20, 37, 38, 46, 51, 57, 64, 67, 71, 78, 88, 90, 91, 100])#1:num_samples
        filename = "rfp_$(ii)_sol.jld2"
        goop_data = load_object("data/relaxably_feasible/GOOP_solution/$filename")
        for jj in 1:num_penalty
            push!(baseline_data, load_object("data/relaxably_feasible/Baseline_solution/$jj/$filename"))
        end

        # Collect difference of slacks for each level
        push!(Δ_slacks_3,
        [sum(baseline_data[jj]["slacks"][1:2:end]) - sum(goop_data["slacks"][1:2:end]) for jj in 1:num_penalty])
        push!(Δ_slacks_2,
        [sum(baseline_data[jj]["slacks"][2:2:end]) - sum(goop_data["slacks"][2:2:end]) for jj in 1:num_penalty])

        # Collect goop and baseline slacks
        push!(goop_slack, goop_data["slacks"])
        push!(baseline_slack, [baseline_data[jj]["slacks"] for jj in 1:num_penalty])

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

        # Initialize baseline_data for next round
        baseline_data = Dict[]
    end

    ## Plot Monte Carlo 1
    n_samples = length(Δ_slacks_3)
    categories = vcat([1:num_penalty for i in 1:n_samples]...)
    values_slack3 = vcat(Δ_slacks_3...)
    values_slack2 = vcat(Δ_slacks_2...)
    colormap = Makie.Categorical(:viridis)
    colors_mc = vcat([[[colormap[25*i] for i in 1:num_penalty]...] for _ in 1:n_samples]...)

    desired_size = [13, 9.2] #cm, double column
    fig = CairoMakie.Figure(size = (desired_size[1]*cm_to_pt, desired_size[2]*cm_to_pt))
    xticks = (0:2:num_penalty, vcat(["G"], ["B$i" for i in 2:2:num_penalty]))
    ax1 = CairoMakie.Axis(fig[1, 1], xticks = xticks, yticks = WilkinsonTicks(4), limits = (nothing, nothing, -0.1, 0.25), xlabel="Method", title=L"\Delta \sum \textbf{s}^i_3")
    violin!(ax1, categories, values_slack3, show_median = true, color=colors_mc)

    ax2 = CairoMakie.Axis(fig[1, 2], xticks = xticks, yticks = WilkinsonTicks(4), xlabel="Method", title=L"\Delta \sum \textbf{s}^i_2")
    violin!(ax2, categories, values_slack2, show_median = true, color=colors_mc)

    CairoMakie.save("./data/relaxably_feasible/result_plots/[MC] rfp_GOOP_Baseline_Monte_Carlo1" * ".pdf", fig, pt_per_unit = 1)

    # Using rainclouds
    desired_size = [18.2, 9.2] #cm, double column
    fig = CairoMakie.Figure(size = (desired_size[1]*cm_to_pt, desired_size[2]*cm_to_pt))
    rainclouds!(CairoMakie.Axis(fig[1,1], title=L"\Delta \sum \textbf{s}^i_3"), categories, values_slack3; cloud_width=1.2, boxplot_width=0.2, side=:right, violin_limits=extrema, color=colors_mc)
    rainclouds!(CairoMakie.Axis(fig[1,2], title=L"\Delta \sum \textbf{s}^i_2"), categories, values_slack2; cloud_width=1.2, boxplot_width=0.2, side=:right, violin_limits=extrema, color=colors_mc)
    CairoMakie.save("./data/relaxably_feasible/result_plots/[MC] rfp_GOOP_Baseline_Monte_Carlo2" * ".pdf", fig, pt_per_unit = 1)

    # Plot Monte Carlo 3 using goop_slack and baseline_slack
    transparency = 0.5
    colors_mc = vcat([[[(colormap[25*i], transparency) for i in 1:num_penalty]...] for _ in 1:n_samples]...)
    desired_size = [20, 10] #cm, double column
    fig = CairoMakie.Figure(size = (desired_size[1]*cm_to_pt, desired_size[2]*cm_to_pt))
    ax1 = CairoMakie.Axis(fig[1, 1], xtickformat = "{:.1f}", xticks = WilkinsonTicks(3), yticks = WilkinsonTicks(3), xlabel=L"\textbf{s}^1_3", ylabel=L"\textbf{s}^1_2", title="Player 1")
    ax2 = CairoMakie.Axis(fig[1, 2], xtickformat = "{:.1f}", ytickformat = "{:.1f}",
        limits = (-0.1, 0.1, 0.6, 0.8),
        xticks = WilkinsonTicks(2), yticks = WilkinsonTicks(2),
        xlabel=L"\textbf{s}^2_3", ylabel=L"\textbf{s}^2_2", title="Player 2")
    ax3 = CairoMakie.Axis(fig[1, 3], xtickformat = "{:.1f}", ytickformat = "{:.1f}",
        limits = (-0.1, 0.1, 0.3, 0.5),
        xticks = WilkinsonTicks(2), yticks = WilkinsonTicks(2),
        xlabel=L"\textbf{s}^3_3", ylabel=L"\textbf{s}^3_2", title="Player 3")
    
    # Plot baseline
    b_player1 = [(baseline_slack[ii][jj][1], baseline_slack[ii][jj][2]) for ii in 1:n_samples for jj in 1:num_penalty]
    CairoMakie.scatter!(ax1, Point2f.(b_player1), color=colors_mc)
    b_player2 = [(baseline_slack[ii][jj][3], baseline_slack[ii][jj][4]) for ii in 1:n_samples for jj in 1:num_penalty]
    CairoMakie.scatter!(ax2, Point2f.(b_player2), color=colors_mc)
    b_player3 = [(baseline_slack[ii][jj][5], baseline_slack[ii][jj][6]) for ii in 1:n_samples for jj in 1:num_penalty]
    CairoMakie.scatter!(ax3, Point2f.(b_player3), color=colors_mc)

    # Plot Goop
    g_player1 = [(goop_slack[ii][1], goop_slack[ii][2]) for ii in 1:n_samples]
    CairoMakie.scatter!(ax1, Point2f.(g_player1), color=(:blue, 0.5))
    g_player2 = [(goop_slack[ii][3], goop_slack[ii][4]) for ii in 1:n_samples]
    CairoMakie.scatter!(ax2, Point2f.(g_player2), color=(:blue, 0.5))
    g_player3 = [(goop_slack[ii][5], goop_slack[ii][6]) for ii in 1:n_samples]
    CairoMakie.scatter!(ax3, Point2f.(g_player3), color=(:blue, 0.5))
    rowgap!(fig.layout, 5)
    CairoMakie.save("./data/relaxably_feasible/result_plots/[MC] rfp_GOOP_Baseline" * ".pdf", fig, pt_per_unit = 1)

    # Plot Monte Carlo 3
    # Main.@infiltrate
    for ii in 1:n_samples
        goop_slack[ii][goop_slack[ii] .< 0] .= 0.0
    end
    colors_mc = vcat([[[colormap[25*i] for i in 1:num_penalty]...] for _ in 1:n_samples]...)
    delta_slack_player1_level3 = [baseline_slack[ii][jj][1]-goop_slack[ii][1] for ii in 1:n_samples for jj in 1:num_penalty]
    delta_slack_player1_level2 = [baseline_slack[ii][jj][2]-goop_slack[ii][2] for ii in 1:n_samples for jj in 1:num_penalty]
    delta_slack_player2_level3 = [baseline_slack[ii][jj][3]-goop_slack[ii][3] for ii in 1:n_samples for jj in 1:num_penalty]
    delta_slack_player2_level2 = [baseline_slack[ii][jj][4]-goop_slack[ii][4] for ii in 1:n_samples for jj in 1:num_penalty]
    delta_slack_player3_level3 = [baseline_slack[ii][jj][5]-goop_slack[ii][5] for ii in 1:n_samples for jj in 1:num_penalty]
    delta_slack_player3_level2 = [baseline_slack[ii][jj][6]-goop_slack[ii][6] for ii in 1:n_samples for jj in 1:num_penalty]

    desired_size = [18.2, 9.2] #cm, double column
    fig = CairoMakie.Figure(size = (desired_size[1]*cm_to_pt, desired_size[2]*cm_to_pt))
    rainclouds!(CairoMakie.Axis(fig[1,1], title=L"\tilde{s}^1_3 - s^1_3"), categories, delta_slack_player1_level3; cloud_width=1.2, boxplot_width=0.2, side=:right, color=colors_mc)
    rainclouds!(CairoMakie.Axis(fig[1,2], title=L"\sum \tilde{s}^1_2 - s^1_2"), categories, delta_slack_player1_level2; cloud_width=1.2, boxplot_width=0.2, side=:right, color=colors_mc)
    CairoMakie.save("./data/relaxably_feasible/result_plots/[MC] rfp_GOOP_Baseline_Monte_Carlo3_player1" * ".pdf", fig, pt_per_unit = 1)

    Main.@infiltrate

    fig = CairoMakie.Figure(size = (desired_size[1]*cm_to_pt, desired_size[2]*cm_to_pt))
    rainclouds!(CairoMakie.Axis(fig[1,1], title=L"\sum \tilde{s}^2_3 - s^2_3"), categories, delta_slack_player2_level3; cloud_width=1.2, boxplot_width=0.2, side=:right, color=colors_mc)
    rainclouds!(CairoMakie.Axis(fig[1,2], title=L"\tilde{s}^2_2 - s^2_2"), categories, delta_slack_player2_level2; cloud_width=1.2, boxplot_width=0.2, side=:right, color=colors_mc)
    CairoMakie.save("./data/relaxably_feasible/result_plots/[MC] rfp_GOOP_Baseline_Monte_Carlo3_player2" * ".pdf", fig, pt_per_unit = 1)
    
    fig = CairoMakie.Figure(size = (desired_size[1]*cm_to_pt, desired_size[2]*cm_to_pt))
    rainclouds!(CairoMakie.Axis(fig[1,1], title=L"\sum \tilde{s}^3_3 - s^3_3"), categories, delta_slack_player3_level3; cloud_width=1.2, boxplot_width=0.2, side=:right, color=colors_mc)
    rainclouds!(CairoMakie.Axis(fig[1,2], title=L"\tilde{s}^3_2 - s^3_2"), categories, delta_slack_player3_level2; cloud_width=1.2, boxplot_width=0.2, side=:right, color=colors_mc)
    CairoMakie.save("./data/relaxably_feasible/result_plots/[MC] rfp_GOOP_Baseline_Monte_Carlo3_player3" * ".pdf", fig, pt_per_unit = 1)
    
    # TODO: Plot Monte Carlo 4 for maximum violation
end


# Plot
plot_goop_vs_penalty(;num_samples=100, num_penalty=10)



end