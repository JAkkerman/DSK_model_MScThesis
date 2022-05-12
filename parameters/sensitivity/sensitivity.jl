"""
This file contains code used to conduct global sensitivity analysis.
The used method is PAWN (https://www.safetoolbox.info/pawn-method/), 
    which is implemented in Python.
"""

using PyCall

include("../../model/main.jl")


"""
Generates data used by sensitivity analysis.
    Code augmented from SAFE package preparation of variables.
"""
function generate_data(
    X_labels::Dict,
    path::String;
    N=100::Int
    )

    # Set up which parameter to apply GSA to, and the range of the parameter

    # Number of uncertain parameters
    M = length(X_labels)

    # Define parameter distributions
    samp_strat = "lhs"

    # Call SAMP function to get the input parameters for running the models
    X = py"call_AAT_sampling"(samp_strat, M, X_labels, N)
    
    # Run simulations for the given parameter values
    Yg_mean = zeros(N)
    Yg_std = zeros(N)

    Cg_mean = zeros(N)
    Cg_std = zeros(N)

    for i in 1:N

        println("Simulation nr: $i")

        # Retrieve variables, repackage to pass to simulation
        Xi = X[i,:]
        changed_params = Dict(k=>Xi[i] for (i,k) in enumerate(keys(X_labels)))

        # Run simulation
        Yg, Cg = run_simulation(
                    changed_params=changed_params,
                    full_output=false
                )

        # Store output data
        Yg_mean[i] = mean(Yg[100:end])
        Yg_std[i] = std(Yg[100:end])

        Cg_mean[i] = mean(Cg[100:end])
        Cg_std[i] = std(Cg[100:end])
    end

    # Set up dataframe containing results
    df = DataFrame(Dict(x=>X[:,i] for (i,x) in enumerate(keys(X_labels))))
    df[!, "Yg_mean"] = Yg_mean
    df[!, "Yg_std"] = Yg_std
    df[!, "Cg_mean"] = Cg_mean
    df[!, "Cg_std"] = Cg_std

    # Write results to csv
    CSV.write(path, df)

end


"""
Calls SAFE toolbox in Python script.
"""
function run_PAWN(
    X_labels::Dict,
    path::String,
    run_nr;
    N=100
    )

    # Read simulation data, save X and Y as matrices
    X = zeros(N, length(X_labels))
    df = DataFrame(CSV.File(path))

    for (i,label) in enumerate(keys(X_labels))
        X[:,i] = df[!, Symbol(label)]
    end

    labels = collect(keys(X_labels))

    Yg_mean = 100 .* df[!, Symbol("Yg_mean")]
    py"run_PAWN"(labels, X, Yg_mean, "Yg_mean", run_nr, "mean GDP growth")

    Yg_std = df[!, Symbol("Yg_std")]
    py"run_PAWN"(labels, X, Yg_std, "Yg_std", run_nr, "std GDP growth")

    Cg_mean = df[!, Symbol("Cg_mean")]
    py"run_PAWN"(labels, X, Cg_mean, "Cg_mean", run_nr, "mean C growth")

    Cg_std = df[!, Symbol("Cg_std")]
    py"run_PAWN"(labels, X, Cg_std, "Cg_std", run_nr, "std C growth")
end


# Include Python file containing GSA functions
@pyinclude("parameters/sensitivity/run_GSA.py")

run_nr = 4

path = "parameters/sensitivity/sensitivity_runs/sensitivity_run_$(run_nr).csv"

N = 8

X_labels = Dict([["α_cp", [0.6, 1.0]],
                 ["μ1", [0.0, 0.4]],
                 ["ω", [0.0, 1.0]],
                 ["ϵ", [0.0, 0.1]],
                 ["κ_upper", [0.0, 0.07]],
                 ["κ_lower", [-0.07, 0.0]],
                 ["ψ_E", [0.0, 1.0]],
                 ["ψ_Q", [0.0, 1.0]],
                 ["ψ_P", [0.0, 1.0]]])

generate_data(X_labels, path; N=N)
# run_PAWN(X_labels, path, run_nr; N=N)