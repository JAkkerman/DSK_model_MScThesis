using CairoMakie
using DataFrames
using CSV


function compute_indexnumbers(ts)
    return ts ./ ts[begin] * 100
end


"""
Plots macroeconomic aggregated variables.
"""
function plot_macro_variables(df::DataFrame, t_warmup)

    f = Figure(resolution=(1000, 1200))
    
    # Income data
    ax = Axis(f[1,1], title="Income indices", ylabel="index")
    lines!(ax, compute_indexnumbers(df[t_warmup:end, :total_real_GDP]), label="total GDP")
    lines!(ax, compute_indexnumbers(df[t_warmup:end, :total_income_hh] ./ df[t_warmup:end, :avg_cp_price]), 
                                    label="household GDP")
    lines!(ax, compute_indexnumbers(df[t_warmup:end, :total_profits_cp] ./ df[t_warmup:end, :avg_cp_price]), 
                                    label="cp GDP")
    lines!(ax, compute_indexnumbers(df[t_warmup:end, :total_profits_kp] ./ df[t_warmup:end, :avg_cp_price]), 
                                    label="kp GDP")
    hlines!(ax, 100., color="black")
    axislegend(ax)

    # Spending data
    ax = Axis(f[1,2], title="Spending indices", ylabel="index")
    lines!(ax, compute_indexnumbers(df[t_warmup:end, :total_investments]), label="I")
    lines!(ax, compute_indexnumbers(df[t_warmup:end, :total_actual_consumption]), label="C")
    hlines!(ax, 100., color="black", alpha=0.2)
    axislegend()

    # Money supply
    ax = Axis(f[2,1], title="Money supply", ylabel="M_{i}")
    lines!(ax, df[!, :M_tot] .- df[!, :debt_tot], label="total")
    lines!(ax, df[!, :M_hh], label="hh")
    lines!(ax, df[!, :M_cp], label="cp")
    lines!(ax, df[!, :M_kp], label="kp")
    axislegend()

    # Price indices
    ax = Axis(f[2,2], title="Price indices", ylabel="index")
    lines!(ax, df[!, :CPI_cp], label="cp")
    lines!(ax, df[!, :CPI_kp], label="kp")
    axislegend()

    # C and I as fraction of GDP
    ax = Axis(f[3,1], title="C and I as fraction of GDP", ylabel="share")
    lines!(ax, df[!, :total_actual_consumption] ./ df[!, :total_nominal_GDP], label="C/GDP")
    lines!(ax, df[!, :total_investments] ./ df[!, :total_nominal_GDP], label="I/GDP")
    axislegend()

    # Annual growth rates
    ax = Axis(f[3,2], title="annual growth rates")
    hist!(ax, df[!, :annual_GDP_growth]; normalization=:probability, bins=50)

    # Unemployment rate
    ax = Axis(f[4,1], title="Unemployment rate", ylabel="rate")
    lines!(ax, df[!, :unemployment_rate], label="unemployment")
    lines!(ax, df[!, :labor_switch_rate], label="switching")
    axislegend()

    # Offered and demanded labor
    ax = Axis(f[4,2], title="Offered and demanded labor", ylabel="number of labor units")
    lines!(ax, df[!, :labor_offered], label="offered")
    lines!(ax, df[!, :labor_demanded], label="demanded")
    lines!(ax, df[!, :labor_hired], label="hired")
    axislegend()
    
    save("plotting/plots/macro_vars_1234.png", f)
end


"""
Plots household-level aggregate data
"""
function plot_household_vars(df::DataFrame, t_warmup)

    f = Figure(resolution=(1000, 800))

    # Wage levels
    ax = Axis(f[1,1], title="Real wage level")
    lines!(ax, df[!, :avg_wage] ./ df[!, :avg_cp_price], label="real wage")
    lines!(ax, df[!, :avg_wage], label="nominal wage")
    axislegend()

    # Household income sources
    ax = Axis(f[1,2], title="Household income sources as share of total income")
    avg_total_income = (
        df[!, :avg_labor_income_hh] 
        .+ df[!, :avg_capital_income_hh]
        .+ df[!, :avg_ub_income_hh]
        .+ df[!, :avg_socben_income_hh]
    )
    lines!(ax, df[!, :avg_labor_income_hh] ./ avg_total_income, label="labor")
    lines!(ax, df[!, :avg_capital_income_hh] ./ avg_total_income, label="capital")
    lines!(ax, df[!, :avg_ub_income_hh] ./ avg_total_income, label="UB")
    lines!(ax, df[!, :avg_socben_income_hh] ./ avg_total_income, label="socben")
    axislegend()

    # Household savings rate
    ax = Axis(f[2,1], title="Savings rate")
    lines!(ax, df[100:end, :savings_total], label="all", color=(:blue, 0.5))
    lines!(ax, df[100:end, :savings_employed], label="employed", color=(:green, 0.2))
    lines!(ax, df[100:end, :savings_unemployed], label="unemployed", color=(:red, 0.2))
    axislegend()

    # Unsatisfied demand
    ax = Axis(f[2,2], title="Unsatisfied demand")
    lines!(ax, df[!, :unsat_consumer_demand], label="D")
    axislegend()

    save("plotting/plots/household_vars_1234.png", f)
end


"""
Plots firm-level aggregate data
"""
function plot_firm_vars(df::DataFrame, t_warmup)

    f = Figure(resolution=(1000, 1800))

    # Labor demand
    ax = Axis(f[1,1], title="labor demand")
    lines!(ax, df[t_warmup:end, :labordemand_change], label="total")
    lines!(ax, df[t_warmup:end, :labordemand_change_cp], label="cp")
    lines!(ax, df[t_warmup:end, :labordemand_change_kp], label="kp")
    axislegend()

    # Debt-to-GDP ratios
    ax = Axis(f[1,2], title="Debt-to-GDP ratio")
    lines!(ax, df[t_warmup:end, :debt_tot] ./ df[t_warmup:end, :total_nominal_GDP], 
           label="total")
    lines!(ax, df[t_warmup:end, :debt_cp] ./ df[t_warmup:end, :total_nominal_GDP], 
           label="cp")
    lines!(ax, df[t_warmup:end, :debt_kp] ./ df[t_warmup:end, :total_nominal_GDP], 
           label="kp")
    axislegend()

    # cp production
    ax = Axis(f[2,1], title="Average consumer good production and demand")
    lines!(ax, df[t_warmup:end, :avg_prod_cp], label="actual Q")
    lines!(ax, df[t_warmup:end, :avg_prod_exp_cp], label="expected Q")
    lines!(ax, df[t_warmup:end, :avg_demand_cp], label="actual D")
    lines!(ax, df[t_warmup:end, :avg_unmet_demand_cp], label="unmet D")
    lines!(ax, df[t_warmup:end, :avg_demand_exp_cp], label="expected D")
    lines!(ax, df[t_warmup:end, :avg_inventories], label="inventories")
    axislegend()

    # kp production
    ax = Axis(f[2,2], title="Total capital good production and demand")
    lines!(ax, df[t_warmup:end, :total_prod_kp], label="actual Q")
    lines!(ax, 25 * (df[t_warmup:end, :total_expand_inv_ordered] .+ df[t_warmup:end, :total_repl_inv_ordered]), 
           label="total ordered")
    lines!(ax, 25 * df[t_warmup:end, :total_expand_inv_ordered], label="EI ordered")
    lines!(ax, 25 * df[t_warmup:end, :total_repl_inv_ordered], label="RS ordered")
    # lines!(ax, df[t_warmup:end, :avg_expans_inv_ordered] .+ df[t_warmup:end, :avg_repl_inv_ordered], label="desired Q")
    axislegend()

    # Unsatisfied demand consumer goods, labor and capital
    ax = Axis(f[3,1], title="Unsatisfied demand")
    lines!(ax, df[t_warmup:end, :unsat_consumer_demand], label="consumer goods")
    lines!(ax, df[t_warmup:end, :unsat_labor_demand], label="labor")
    lines!(ax, df[t_warmup:end, :unsat_capital_demand], label="capital goods")
    axislegend()


    # Average machine demand
    ax = Axis(f[4,1], title="Average machine demand")
    lines!(ax, df[t_warmup:end, :avg_expans_inv_ordered], label="EI ordered")
    lines!(ax, df[t_warmup:end, :avg_repl_inv_ordered], label="RS ordered")
    axislegend()

    # Monhtly banktupty rates
    ax = Axis(f[4,2], title="Montly bankrupty rates")
    lines!(ax, df[t_warmup:end, :bankrupty_rate_cp], label="cp")
    lines!(ax, df[t_warmup:end, :bankrupty_rate_kp], label="kp")
    axislegend()

    # Technology levels
    ax = Axis(f[5,1], title="Technology levels")
    lines!(ax, df[!, :avg_actual_tech_labor], label="labor")
    lines!(ax, df[!, :avg_actual_tech_energy], label="energy")
    lines!(ax, df[!, :avg_actual_tech_envir], label="environmental")
    axislegend()

    # Markups
    ax = Axis(f[5,2], title="Markups")
    lines!(ax, df[!, :markup_cp], label="cp")
    lines!(ax, df[!, :markup_kp], label="kp")
    axislegend()

    save("plotting/plots/firm_vars_1234.png", f)
end


function plot_gov_vars(df::DataFrame, t_warmup)

    f = Figure(resolution=(1000, 800))

    # Wage levels
    ax = Axis(f[1,1], title="Tax revenues")
    lines!(ax, df[!, :avg_wage] ./ df[!, :avg_cp_price], label="real wage")
    lines!(ax, df[!, :avg_wage], label="nominal wage")
    axislegend()

end


function plot_output_single_run(;t_warmup=300)
    df = DataFrame(CSV.File("results/result_data/model_data_1234.csv"))

    # Plot time series of macroeconomic variables
    plot_macro_variables(df, t_warmup)

    # Plot household variables
    plot_household_vars(df, t_warmup)

    # Plot firm variables
    plot_firm_vars(df, t_warmup)

    # Plot government variables
    # plot_gov_vars(df, t_warmup)
end
plot_output_single_run()