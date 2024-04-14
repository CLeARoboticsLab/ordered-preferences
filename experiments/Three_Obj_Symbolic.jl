module Three_Obj_Symbolic


using OrderedPreferences
using Makie, CairoMakie

function demo(;verbose = false)

    # Lex Min QP Fiaschi (2021)
    # Three objectives:
    c = [-1; -1; -1]
    Q = [2 2 0; 
          2 2 0; 
          0 0 4]  
    q = [-5; -5; 0]
    P = [4 0 0; 
          0 4 0; 
          0 0 0]  
    p = [-5; -3; 2]

    # Define objective and constraints
    objective(x,θ) = 0.5*x[1:3]'P*x[1:3] + p'x[1:3] #2*(x[1]^2 + x[2]^2)

    c1(x,θ) =  x[1] - x[2] - x[3] + 1
    c2(x,θ) =  x[1] + x[2] - x[3] + 1
    c3(x,θ) = -x[1] + x[2] - x[3] + 1
    c4(x,θ) = -x[1] - x[2] - x[3] + 3
    c5(x,θ) =  x[3] 
    inequality_constraints(x,θ) = [
        c1(x,θ);
        c2(x,θ);
        c3(x,θ);
        c4(x,θ);
        c5(x,θ);
    ] 

    equality_constraints(x,θ) = []

    prioritized_preferences = [
        function (x, θ)
            c'x[1:3]
        end,
        
        function (x, θ)
            0.5*x[1:3]'Q*x[1:3] + q'x[1:3]
        end,
    ]

    # Problem setting 
    primal_dimension = 3
    parameter_dimension = 2 # stay user-defined
    parameters = ones(parameter_dimension) # dummy 

    equality_dimension = length(equality_constraints(zeros(primal_dimension), zeros(parameter_dimension)))
    inequality_dimension = length(inequality_constraints(zeros(primal_dimension), zeros(parameter_dimension)))
    
    # Algorithm setting 
    ϵ = 1.0
    κ = 0.1
    max_iterations = 10
    tolerance = 1e-7
    relaxation_mode = :l_infinity
    println("relaxation_mode: ", relaxation_mode)

    POP_prob = ParametricOrderedPreferencesMPCC(; # Stay parametrized by θ where θ(end) is the relaxation parameter
        objective,
        equality_constraints,
        inequality_constraints,
        prioritized_preferences,
        primal_dimension,
        parameter_dimension,
        equality_dimension,
        inequality_dimension,
        relaxation_mode,
    )

    # Solve POP
    (; relaxation, solution, residual) = 
        solve_relaxed_pop(POP_prob, nothing, parameters; ϵ, κ, max_iterations, tolerance, verbose)
    println("relaxation: ", relaxation)
    println("solution: ", solution[end]) #TODO: solution
    println("residual: ", residual)

    # Plot convergence 
    colors = [:blue, :green, :red, :brown, :purple]
    index = CairoMakie.LinRange(1:length(solution))
    x = Vector{Float64}[]
    λ1 = Vector{Float64}[]
    λ2 = Vector{Float64}[]
    μ = Vector{Float64}[]
    for sol_ii in solution 
        push!(x, sol_ii.primals[1:3])
        push!(λ1, sol_ii.primals[4:8])
        push!(λ2, sol_ii.primals[9:19])
        push!(μ, sol_ii.primals[20:22])    
    end 
    # 1. Plot x[1], x[2], x[3]
    fig = Figure(size = (1200, 900)) # 2400x1800 pixels
    ax1 = Axis(fig[1, 1], xlabel = "Iteration", xticks = index, title = "Convergence of x₁, x₂, x₃")
    for jj in 1:3
        scatterlines!(ax1, index, [x[ii][jj] for ii in index], color = colors[jj], markercolor = colors[jj], label = "x$jj")
    end
    axislegend(position = :rc)

    # 2. Plot (exact) objective
    ax2 = Axis(fig[1, 2], xlabel = "Iteration", xticks = index, title = "Convergence of objective")
    scatterlines!(ax2, index, [objective(x[ii],0) for ii in index], color = :black, markercolor = :black, label = "objective")
    axislegend(position = :rc)

    # 3. Plot (exact) complementarity violations
    ax3 = Axis(fig[2, 1], xlabel = "Iteration", xticks = index)
    for jj in 1:5
        scatterlines!(ax3, index, [λ1[ii][jj] * λ2[ii][jj+5] for ii in index], color = colors[jj], markercolor = colors[jj], label = L"\lambda^{1}_{%$(jj)} \lambda^{2}_{%$(jj+5)}")
    end
    axislegend(position = :rc, orientation = :horizontal, tellwidth = false, tellheight = true)

    ax4 = Axis(fig[2, 2], xlabel = "Iteration", xticks = index)
    scatterlines!(ax4, index, [c1(x[ii],0) * λ2[ii][1] for ii in index], color = :blue, markercolor = :blue, label = L"\lambda^{2}_{1} c_{1}(x)")
    scatterlines!(ax4, index, [c2(x[ii],0) * λ2[ii][2] for ii in index], color = :green, markercolor = :green, label = L"\lambda^{2}_{2} c_{2}(x)")
    scatterlines!(ax4, index, [c3(x[ii],0) * λ2[ii][3] for ii in index], color = :red, markercolor = :red, label = L"\lambda^{2}_{3} c_{3}(x)")
    scatterlines!(ax4, index, [c4(x[ii],0) * λ2[ii][4] for ii in index], color = :brown, markercolor = :brown, label = L"\lambda^{2}_{4} c_{4}(x)")
    scatterlines!(ax4, index, [c5(x[ii],0) * λ2[ii][5] for ii in index], color = :purple, markercolor = :purple, label = L"\lambda^{2}_{5} c_{5}(x)")
    axislegend(position = :rt, orientation = :horizontal, tellwidth = false, tellheight = true)

    ax5 = Axis(fig[3, 1], xlabel = "Iteration", xticks = index)
    scatterlines!(ax5, index, [-(λ1[ii][1]*c1(x[ii],0) + λ1[ii][2]*c2(x[ii],0) + λ1[ii][3]*c3(x[ii],0) + λ1[ii][4]*c4(x[ii],0) + λ1[ii][5]*c5(x[ii],0)) * λ2[ii][11] for ii in index], color = :blue, markercolor = :blue, label = L"-\lambda^{2}_{11} \sum_{i=1}^{5}c_{i}(x) ")
    axislegend(position = :rc, orientation = :horizontal, tellwidth = false, tellheight = true)

    ax6 = Axis(fig[3, 2], xlabel = "Iteration", xticks = index)
    scatterlines!(ax6, index, [c1(x[ii],0) * λ1[ii][1] for ii in index], color = :blue, markercolor = :blue, label = L"\lambda^{1}_{1} c_{1}(x)")
    scatterlines!(ax6, index, [c2(x[ii],0) * λ1[ii][2] for ii in index], color = :green, markercolor = :green, label = L"\lambda^{1}_{2} c_{2}(x)")
    scatterlines!(ax6, index, [c3(x[ii],0) * λ1[ii][3] for ii in index], color = :red, markercolor = :red, label = L"\lambda^{1}_{3} c_{3}(x)")
    scatterlines!(ax6, index, [c4(x[ii],0) * λ1[ii][4] for ii in index], color = :brown, markercolor = :brown, label = L"\lambda^{1}_{4} c_{4}(x)")
    scatterlines!(ax6, index, [c5(x[ii],0) * λ1[ii][5] for ii in index], color = :purple, markercolor = :purple, label = L"\lambda^{1}_{5} c_{5}(x)")
    axislegend(position = :rt, orientation = :horizontal, tellwidth = false, tellheight = true)

    display(fig)
end


end # module end