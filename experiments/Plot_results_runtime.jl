module Plot_results_runtime

using JLD2, CairoMakie, ProgressMeter

function plot_runtime(;num_samples = 100, num_penalty = 6)

    categories = ["baseline", "hybrid", "goop"]
    baseline_runtime_data = Float64[]
    not_include = [23, 24, 25, 26, 27, 28]
    for ii in setdiff(1:num_samples, not_include)
        # Load Baseline runtime data
        baseline_data = push!(baseline_runtime_data, load_object("./data/relaxably_feasible/runtime/rfp_runtime_$(ii)_baseline.jld2")...)
    end
    # Load Hybrid runtime data
    hybrid_data = load_object("./data/relaxably_feasible/runtime/rfp_runtime_hybrid.jld2")
        
    # Load GOOP runtime data
    goop_data = load_object("./data/relaxably_feasible/runtime/rfp_runtime_goop.jld2")

    # Remove unconverged data
    hybrid_data = hybrid_data[[i for i in 1:length(hybrid_data) if !(i in not_include)]]
    goop_data = goop_data[[i for i in 1:length(goop_data) if !(i in not_include)]]

    Main.@infiltrate
    aggregate_data = vcat(baseline_runtime_data, hybrid_data, goop_data)
    aggregate_categories = mapreduce(vcat, 1:length(categories)) do k
        vcat([categories[k] for _ in 1:length(baseline_runtime_data)])
    end

    Main.@infiltrate

    
end

# Plot
plot_runtime(;num_samples=100, num_penalty=6)


end