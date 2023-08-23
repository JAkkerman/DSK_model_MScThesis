# TODO: only macro variables required for functioning model, others in general data collector

@Base.kwdef mutable struct MacroEconomy
    T::Int=T                                                # number of timesteps

    # GDP
    GDP_nominal::Vector{Float64} = zeros(Float64, T)        # nominal GDP over time
    GDP_real::Vector{Float64} = zeros(Float64, T)           # real GDP over time
    GDP_hh::Vector{Float64} = zeros(Float64, T)             # household income share of GDP over time
    GDP_cp::Vector{Float64} = zeros(Float64, T)             # cp profit share of GDP over time
    GDP_kp::Vector{Float64} = zeros(Float64, T)             # kp profit share of GDP over time

    GDP_qrt_growth::Vector{Float64} = zeros(Float64, T)     # quarterly GDP growth rates
    GDP_ann_growth::Vector{Float64} = zeros(Float64, T)     # annual GDP growth rates

    # total_C::Vector{Float64} = zeros(Float64, T)            # total spending on consumption
    # total_C_actual::Vector{Float64} = zeros(Float64, T)     # total actual spending on consumption
    
    # total_I::Vector{Float64} = zeros(Float64, T)            # total actual spending on investments
    # total_w::Vector{Float64} = zeros(Float64, T)            # total actual spending on wages
    # LIS::Vector{Float64} = zeros(Float64, T)                # labor income share

    returns_investments::Vector{Float64} = zeros(Float64, T)# returns to investments

    p_avg_cp::Vector{Float64} = zeros(Float64, T)           # average price of cp goods over time
    p_avg_kp::Vector{Float64} = zeros(Float64, T)           # average price of kp goods over time
    # markup_cp::Vector{Float64} = zeros(Float64, T)          # average μ for cp
    # markup_kp::Vector{Float64} = zeros(Float64, T)          # average μ for kp
    # CPI_cp::Vector{Float64} = zeros(Float64, T)             # price index consumer goods over time
    # CPI_kp::Vector{Float64} = zeros(Float64, T)             # price index capital goods over time

    # unsat_demand::Vector{Float64} = zeros(Float64, T)       # ratio of unsatisfied demand
    # unsat_invest::Vector{Float64} = zeros(Float64, T)       # ratio of unsatisfied investments
    # unsat_L_demand::Vector{Float64} = zeros(Float64, T)     # ratio of unsatisfied labor demand
    # unspend_C::Vector{Float64} = zeros(Float64, T)          # average ratio of unspend C
    # N_goods::Vector{Float64} = zeros(Float64, T)            # total amount of inventories
    # avg_N_goods::Vector{Float64} = zeros(Float64, T)        # average number of goods in cp firm inventory

    # Division of money over sectors
    M::Vector{Float64} = zeros(Float64, T)                  # total amount of money (should be stable)
    M_hh::Vector{Float64} = zeros(Float64, T)               # total amount of money at hh
    M_cp::Vector{Float64} = zeros(Float64, T)               # total amount of money at cp
    M_kp::Vector{Float64} = zeros(Float64, T)               # total amount of money at kp
    M_ep::Vector{Float64} = zeros(Float64, T)               # total amount of money at ep
    M_gov::Vector{Float64} = zeros(Float64, T)              # total amount of money at gov
    M_if::Vector{Float64}  = zeros(Float64, T)              # total amount of money at if

    # Debt levels
    debt_tot::Vector{Float64} = zeros(Float64, T)           # total debt
    debt_cp::Vector{Float64} = zeros(Float64, T)            # cp debt
    debt_kp::Vector{Float64} = zeros(Float64, T)            # kp debt
    # debt_cp_allowed::Vector{Float64} = zeros(Float64, T)    # cp allowed debt
    # debt_kp_allowed::Vector{Float64} = zeros(Float64, T)    # kp debt allowed
    debt_unpaid_cp::Vector{Float64} = zeros(Float64, T)     # cp debt that went unpaid after bankrupcy
    debt_unpaid_kp::Vector{Float64} = zeros(Float64, T)     # kp debt that went unpaid after bankrupcy

    # Wage statistics
    w_avg::Vector{Float64} = ones(Float64, T)               # average wage over time
    # w_req_avg::Vector{Float64} = ones(Float64, T)           # average requested wage over time
    # w_sat_avg::Vector{Float64} = ones(Float64, T)           # average of satisfying wage over time
    
    # Household incomes and savings
    Y_avg::Vector{Float64} = zeros(Float64, T)              # average income over time
    # YL_avg::Vector{Float64} = zeros(Float64, T)             # average labor income over time
    # YK_avg::Vector{Float64} = zeros(Float64, T)             # average capital income over time
    # YUB_avg::Vector{Float64} = zeros(Float64, T)            # average UB income over time
    # YSB_avg::Vector{Float64} = zeros(Float64, T)            # average social benefits income over time
    # s_emp::Vector{Float64} = zeros(Float64, T)              # average savings rate of employed households over time
    # s_unemp::Vector{Float64} = zeros(Float64, T)            # average savings rate of unemployed households over time

    # Labor market
    U::Vector{Float64} = zeros(Float64, T)                  # unemployment over time
    # switch_rate::Vector{Float64} = zeros(Float64, T)        # rate of switching employers
    # L_offered::Vector{Float64} = zeros(Float64, T)          # labor units offered
    # L_demanded::Vector{Float64} = zeros(Float64, T)         # labor units demanded
    # L_hired::Vector{Float64} = zeros(Float64, T)            # labor units hired
    # dL_avg::Vector{Float64} = zeros(Float64, T)             # average desired labor change
    # dL_cp_avg::Vector{Float64} = zeros(Float64, T)          # average desired over time for cp
    # dL_kp_avg::Vector{Float64} = zeros(Float64, T)          # average desired over time for kp
    
    # Investments
    # RD_total::Vector{Float64} = zeros(Float64, T)           # total R&D investments
    # EI_avg::Vector{Float64}  = zeros(Float64, T)            # average expansion investment
    # n_mach_EI_avg::Vector{Float64} = zeros(Float64, T)      # average amount of ordered machines for EI
    # RS_avg::Vector{Float64} = zeros(Float64, T)             # average replacement investment
    # n_mach_RS_avg::Vector{Float64} = zeros(Float64, T)      # average amounf of ordered machines for RS

    # Productivity
    # avg_pi_LP::Vector{Float64} = zeros(Float64, T)          # average labor productivity cp
    # avg_pi_EE::Vector{Float64} = zeros(Float64, T)          # average productivity per energy unit cp
    # avg_pi_EF::Vector{Float64} = zeros(Float64, T)          # average energy friendliness 

    # avg_A_LP::Vector{Float64} = zeros(Float64, T)           # average A_LP at kp
    # avg_A_EE::Vector{Float64} = zeros(Float64, T)           # average A_EE at kp
    # avg_A_EF::Vector{Float64} = zeros(Float64, T)           # average A_EF at kp

    # avg_B_LP::Vector{Float64} = zeros(Float64, T)           # average B_LP at kp
    # avg_B_EE::Vector{Float64} = zeros(Float64, T)           # average B_EE at kp
    # avg_B_EF::Vector{Float64} = zeros(Float64, T)           # average B_EF at kp

    # Production
    # total_Q_cp::Vector{Float64} = zeros(Float64, T)         # total units produced by cp
    # total_Q_kp::Vector{Float64} = zeros(Float64, T)         # total units produced by kp
    # total_Q_growth::Vector{Float64} = zeros(Float64, T)     # growth in total units produced
    # avg_Q_cp::Vector{Float64} = zeros(Float64, T)           # average production of cp
    # avg_Qs_cp::Vector{Float64} = zeros(Float64, T)          # average desired ST production of cp
    # avg_Qe_cp::Vector{Float64} = zeros(Float64, T)          # average desired LT production of cp
    # avg_Q_kp::Vector{Float64} = zeros(Float64, T)           # average production of kp
    # avg_D_cp::Vector{Float64} = zeros(Float64, T)           # average demand of cp
    # avg_Du_cp::Vector{Float64} = zeros(Float64, T)          # average unsatisfied demand of cp
    # avg_De_cp::Vector{Float64} = zeros(Float64, T)          # average expected demand of cp

    # Bankrupties
    bankrupt_cp::Vector{Float64} = zeros(Float64, T)        # fraction of cp that went bankrupt
    bankrupt_kp::Vector{Float64} = zeros(Float64, T)        # fraction of kp that went bankrupt

    # cu::Vector{Float64} = zeros(Float64, T)                 # average rate of capital utilization
    # avg_n_machines_cp::Vector{Float64} = zeros(Float64, T)  # average number of machines cp

    # GINI_I::Vector{Float64} = zeros(Float64, T)             # Gini coefficient for income
    # GINI_W::Vector{Float64} = zeros(Float64, T)             # Gini coefficient for wealth
end


"""
Determines all macro data to aggregate and collect for final data set.
"""
function aggregate_data(model::ABM)
    data_to_collect = [

        # Income data
        total_nominal_GDP,
        total_real_GDP,
        total_income_hh,
        total_profits_cp,
        total_profits_kp,
        total_profits_ep,
        quarterly_GDP_growth,
        annual_GDP_growth,
        LIS,
        GINI_income,
        GINI_wealth,
        savings_total,
        savings_employed,
        savings_unemployed,

        # Price data
        avg_cp_price,
        avg_kp_price,
        CPI_cp,
        CPI_kp,
        markup_cp,
        markup_kp,

        # Spending data
        planned_consumption,
        actual_consumption,
        unspend_consumption,
        total_investments,
        total_wagespending,
        unsat_consumer_demand,
        unsat_capital_demand,
        unsat_labor_demand,

        # Investments
        RD_total,
        expans_inv_desired_avg,
        expans_inv_ordered_avg,
        repl_inv_desired_avg,
        repl_inv_ordered_avg,

        # Labor market
        labor_offered,
        labor_demanded,
        labor_hired,
        labordemand_change,
        labordemand_change_cp,
        labordemand_change_kp,
        unemployment_rate,
        labor_switch_rate,
        avg_wage,
        avg_req_wage,
        avg_sat_wage,

        # Money supply
        M_hh,
        M_cp,
        M_kp,
        M_ep,
        M_gov,
        M_if,
        M_tot,

        # Debt
        debt_tot,
        debt_cp,
        debt_kp,
        debt_allowed_cp,
        debt_allowed_kp,
        debt_unpaid_cp,
        debt_unpaid_kp,

        # Technology levels
        avg_actual_tech_labor,
        avg_actual_tech_energy,
        avg_actual_tech_envir,
        avg_A_tech_labor,
        avg_A_tech_energy,
        avg_A_tech_envir,
        avg_B_tech_labor,
        avg_B_tech_energy,
        avg_B_tech_envir,

        # Production and demand
        total_prod_cp,
        total_prod_kp,
        avg_prod_cp,
        avg_prod_exp_cp,
        avg_demand_cp,
        avg_unmet_demand_cp,
        avg_exp_demand_cp,
        avg_prod_kp,

        # Bankrupties
        bankrupty_rate_cp,
        bankrupty_rate_kp,

        # Inventories
        total_inventories,
        avg_inventories,

        # Machines
        capital_utilization,
        avg_n_machines
    ]
    return data_to_collect
end


"""
Income data on aggregate and per agent type
"""

# Total household income, and income sources
total_income_hh(model) = model.t > 0 ? model.macroeconomy.GDP_hh[model.t] : 0.
avg_labor_income_hh(model) = mean(hh_id -> model[hh_id].labor_I, model.all_hh)
avg_capital_income_hh(model) = mean(hh_id -> model[hh_id].capital_I, model.all_hh)
avg_ub_income_hh(model) = mean(hh_id -> model[hh_id].UB_I, model.all_hh)
avg_socben_income_hh(model) = mean(hh_id -> model[hh_id].socben_I, model.all_hh)

# Firm profits
total_profits_cp(model) = model.t > 0 ? model.macroeconomy.GDP_cp[model.t] : 0.
total_profits_kp(model) = model.t > 0 ? model.macroeconomy.GDP_kp[model.t] : 0.
total_profits_ep(model) = model.t > 0 ? model.ep.net_profit[model.t] : 0.

# total GDP
total_nominal_GDP(model) = model.t > 0 ? model.macroeconomy.GDP_nominal[model.t] : 0.
total_real_GDP(model) = model.t > 0 ? model.macroeconomy.GDP_real[model.t] : 0.
quarterly_GDP_growth(model) = model.t > 0 ? model.macroeconomy.GDP_qrt_growth[model.t] : 0.
annual_GDP_growth(model) = model.t > 0 ? model.macroeconomy.GDP_ann_growth[model.t] : 0.

# Compute the labor share of income
LIS(model) = (model.t > 0 ? sum(p_id -> model[p_id].curracc.TCL, model.all_p) 
                                / model.macroeconomy.GDP_nominal[model.t] : 0.)

# Compute GINI coefficients
GINI_income(model) = GINI_approx(map(hh_id -> model[hh_id].total_I[end], model.all_hh), model)
GINI_wealth(model) = GINI_approx(map(hh_id -> model[hh_id].W, model.all_hh), model)

# Compute return data
returns_investments(model) = model.idxf.returns_investments

# Household savings
savings_total(model) = mean(hh_id -> model[hh_id].s, model.all_hh)
savings_employed(model) = (length(model.labormarket.employed_hh) > 1 ?
    mean(hh_id -> model[hh_id].s, model.labormarket.employed_hh) : NaN
)
savings_unemployed(model) = (length(model.labormarket.unemployed_hh) > 1 ?
    mean(hh_id -> model[hh_id].s, model.labormarket.unemployed_hh) : NaN
)


"""
Price data
"""

# Compute average price, weighted by market share (just retrieve from macro struct)
avg_cp_price(model) = model.t > 0 ? model.macroeconomy.p_avg_cp[model.t] : 0.
avg_kp_price(model) = model.t > 0 ? model.macroeconomy.p_avg_kp[model.t] : 0.
CPI_cp(model) = model.t > 1 ? 100. / model.macroeconomy.p_avg_cp[begin] * model.macroeconomy.p_avg_cp[model.t] : 100.
CPI_kp(model) = model.t > 1 ? 100. / model.macroeconomy.p_avg_kp[begin] * model.macroeconomy.p_avg_kp[model.t] : 100.

# Update markup rates
markup_cp(model) = mean(cp_id -> model[cp_id].μ[end], model.all_cp)
markup_kp(model) = mean(kp_id -> model[kp_id].μ[end], model.all_kp)


""" 
Agent spending data during period
"""

# Household consumption
planned_consumption(model) = sum(hh_id -> model[hh_id].C, model.all_hh)
actual_consumption(model) = sum(hh_id -> model[hh_id].C_actual, model.all_hh)
unspend_consumption(model) = (1 - sum(hh_id -> model[hh_id].C_actual, model.all_hh) 
                              / sum(hh_id -> model[hh_id].C, model.all_hh))

total_investments(model) = sum(cp_id -> model[cp_id].curracc.TCI, model.all_cp)
total_wagespending(model) = sum(p_id -> model[p_id].curracc.TCL, model.all_p)

# Unsatisfied demand in consumer good, capital good and labor markets
unsat_consumer_demand(model) = (sum(cp_id -> model[cp_id].Dᵁ[end], model.all_cp) 
                        / sum(cp_id -> model[cp_id].D[end] + model[cp_id].Dᵁ[end], model.all_cp))
unsat_capital_demand(model) = (1 - sum(kp_id -> model[kp_id].Q[end], model.all_kp) 
                        / (sum(cp_id -> model.g_param.freq_per_machine * (model[cp_id].n_mach_ordered_EI + model[cp_id].n_mach_ordered_RS), model.all_cp)))
unsat_labor_demand(model) = (1 - model.labormarket.L_hired 
                        / model.labormarket.L_demanded)


"""
Labor market and wages
"""

# Labor demand and supply
labor_offered(model) = model.labormarket.L_offered
labor_demanded(model) = model.labormarket.L_demanded
labor_hired(model) = model.labormarket.L_hired

labordemand_change(model) = mean(p_id -> model[p_id].ΔLᵈ, model.all_p)
labordemand_change_cp(model) = mean(cp_id -> model[cp_id].ΔLᵈ, model.all_cp)
labordemand_change_kp(model) = mean(kp_id -> model[kp_id].ΔLᵈ, model.all_kp)

# Unemployment rate and unemployment benefits expenditure
unemployment_rate(model) = model.t > 0 ? model.macroeconomy.U[model.t] : 0.
labor_switch_rate(model) = model.labormarket.switch_rate

# Wage data
avg_wage(model) = model.t > 0 ? model.macroeconomy.w_avg[model.t] : 0.
avg_req_wage(model) = mean(hh_id -> model[hh_id].wʳ, model.all_hh)
avg_sat_wage(model) = mean(hh_id -> model[hh_id].wˢ, model.all_hh)


"""
Money supply
"""

M_hh(model) = model.t > 0 ? model.macroeconomy.M_hh[model.t] : 0.
M_cp(model) = model.t > 0 ? model.macroeconomy.M_cp[model.t] : 0.
M_kp(model) = model.t > 0 ? model.macroeconomy.M_kp[model.t] : 0.
M_ep(model) = model.t > 0 ? model.macroeconomy.M_ep[model.t] : 0.
M_gov(model) = model.t > 0 ? model.macroeconomy.M_gov[model.t] : 0.
M_if(model) = model.t > 0 ? model.macroeconomy.M_if[model.t] : 0.
M_tot(model) = model.t > 0 ? model.macroeconomy.M[model.t] : 0.


"""
Debt levels
"""

debt_tot(model) = model.t > 0 ? model.macroeconomy.debt_tot[model.t] : 0.
debt_cp(model) = model.t > 0 ? model.macroeconomy.debt_cp[model.t] : 0.
debt_kp(model) = model.t > 0 ? model.macroeconomy.debt_kp[model.t] : 0.
debt_allowed_cp(model) = model.g_param.Λ * sum(cp_id -> model[cp_id].curracc.S, model.all_cp)
debt_allowed_kp(model) = model.g_param.Λ * sum(kp_id -> model[kp_id].curracc.S, model.all_kp)
debt_unpaid_cp(model) = model.t > 0 ? model.macroeconomy.debt_unpaid_cp[model.t] : 0.
debt_unpaid_kp(model) = model.t > 0 ? model.macroeconomy.debt_unpaid_kp[model.t] : 0.


"""
Investments
"""

# R&D investments 
RD_total(model) = model.t > 0 ? sum(kp_id -> model[kp_id].RD, model.all_kp) + model.ep.RD_ep[model.t] : 0.

# Desired and ordered machines
expans_inv_desired_avg(model) = mean(cp_id -> model[cp_id].EIᵈ, model.all_cp)
expans_inv_ordered_avg(model) = mean(cp_id -> model[cp_id].n_mach_ordered_EI, model.all_cp)
repl_inv_desired_avg(model) = mean(cp_id -> model[cp_id].RSᵈ, model.all_cp)
repl_inv_ordered_avg(model) = mean(cp_id -> model[cp_id].n_mach_ordered_RS, model.all_cp)


"""
Technology Levels
"""

avg_actual_tech_labor(model) = mean(cp_id -> model[cp_id].π_LP, model.all_cp)
avg_actual_tech_energy(model) = mean(cp_id -> model[cp_id].π_EE, model.all_cp)
avg_actual_tech_envir(model) = mean(cp_id -> model[cp_id].π_EF, model.all_cp)

avg_A_tech_labor(model) = mean(kp_id -> model[kp_id].A_LP[end], model.all_kp)
avg_A_tech_energy(model) = mean(kp_id -> model[kp_id].A_EE[end], model.all_kp)
avg_A_tech_envir(model) = mean(kp_id -> model[kp_id].A_EF[end], model.all_kp)

avg_B_tech_labor(model) = mean(kp_id -> model[kp_id].B_LP[end], model.all_kp)
avg_B_tech_energy(model) = mean(kp_id -> model[kp_id].B_EE[end], model.all_kp)
avg_B_tech_envir(model) = mean(kp_id -> model[kp_id].B_EF[end], model.all_kp)


"""
Production and demand quantities
"""

total_prod_cp(model) = sum(cp_id -> model[cp_id].Q[end], model.all_cp)
total_prod_kp(model) = sum(kp_id -> model[kp_id].Q[end], model.all_kp)

avg_prod_cp(model) = mean(cp_id -> model[cp_id].Q[end], model.all_cp)
avg_prod_exp_cp(model) = mean(cp_id -> model[cp_id].Qᵉ, model.all_cp)
avg_demand_cp(model) = mean(cp_id -> model[cp_id].D[end], model.all_cp)
avg_unmet_demand_cp(model) = mean(cp_id -> model[cp_id].Dᵁ[end], model.all_cp)
avg_exp_demand_cp(model)= mean(cp_id -> model[cp_id].Dᵉ, model.all_cp)

avg_prod_kp(model) = mean(kp_id -> model[kp_id].Q[end], model.all_kp)


"""
Bankrupties
"""
bankrupty_rate_cp(model) = model.t > 0 ? model.macroeconomy.bankrupt_cp[model.t] : 0.
bankrupty_rate_kp(model) = model.t > 0 ? model.macroeconomy.bankrupt_kp[model.t] : 0.


"""
Inventories
"""
total_inventories(model) = sum(cp_id -> model[cp_id].N_goods, model.all_cp)
avg_inventories(model) = mean(cp_id -> model[cp_id].N_goods, model.all_cp)


"""
Machines
"""

capital_utilization(model) = mean(skipmissing(map(cp_id -> model[cp_id].n_machines > 0 ? model[cp_id].cu : missing, model.all_cp)))
avg_n_machines(model) = mean(cp_id -> model[cp_id].n_machines, model.all_cp)


# TODO: MOVE TO WRITEDATA
function get_mdata(model::ABM)::DataFrame

    # If mdata to save is not specified, save all data in macro struct
    macro_categories = fieldnames(typeof(model.macroeconomy))[2:end-1]
    if !isnothing(model.mdata_tosave)
        macro_categories = model.mdata_tosave
    end

    # Gather macro data in dict
    macro_dict = Dict(cat => getproperty(model.macroeconomy, cat) 
                      for cat in macro_categories)

    # Gather energy producer data in dict
    ep_dict = Dict(cat => getproperty(model.ep, cat) for cat in model.epdata_tosave)

    # Gather climate data in dict
    climate_dict = Dict(cat => getproperty(model.climate, cat) for cat in model.climatedata_tosave)

    # Gather government tax revenue and expenditure data
    government_dict = Dict(cat => getproperty(model.gov, cat) for cat in model.governmentdata_tosave)

    # Merge all dicts and convert and return as dataframe
    model_dict = merge(macro_dict, ep_dict, climate_dict, government_dict)
    return DataFrame(model_dict)
end


"""
Calls update functions required for macro variables used in simulation.
"""
function update_macro_timeseries(bankrupt_cp, bankrupt_kp, model::ABM,)

    # Update CPI
    compute_price_data!(model)

    # Update total GDP, per sector and GDP growth
    compute_GDP!(model)

    # Update spending of different sectors
    # compute_spending!(model)


    # Compute the labor share of income
    # model.macroeconomy.LIS[t] = model.macroeconomy.total_w[t] / model.macroeconomy.GDP[t]

    # Update returns to investments
    model.macroeconomy.returns_investments[model.t] = model.idxf.returns_investments

    # # Update labor demand
    # model.macroeconomy.L_offered[t] = model.labormarket.L_offered
    # model.macroeconomy.L_demanded[t] = model.labormarket.L_demanded
    # model.macroeconomy.L_hired[t] = model.labormarket.L_hired

    # model.macroeconomy.dL_avg[t] = mean(p_id -> model[p_id].ΔLᵈ, all_p)
    # model.macroeconomy.dL_cp_avg[t] = mean(cp_id -> model[cp_id].ΔLᵈ, all_cp)
    # model.macroeconomy.dL_kp_avg[t] = mean(kp_id -> model[kp_id].ΔLᵈ, all_kp)

    # # Update unemployment rate and unemployment benefits expenditure
    model.macroeconomy.U[model.t] = model.labormarket.E
    # model.macroeconomy.switch_rate[t] = labormarket.switch_rate

    # Compute total amount in system
    compute_M!(model)

    # Compute average savings rates
    # compute_savings_macro!(model)

    # Wage and income statistics
    update_wage_stats!(model)
    # update_income_stats!(all_hh, t, model)

    update_debt!(bankrupt_cp, bankrupt_kp, model)

    # # Investment
    # model.macroeconomy.RD_total[t] = sum(kp_id -> model[kp_id].RD, all_kp) + ep.RD_ep[t]
    # model.macroeconomy.EI_avg[t] = mean(cp_id -> model[cp_id].EIᵈ, all_cp)
    # model.macroeconomy.n_mach_EI_avg[t] = mean(cp_id -> model[cp_id].n_mach_ordered_EI, all_cp)
    # model.macroeconomy.RS_avg[t] = mean(cp_id -> model[cp_id].RSᵈ, all_cp)
    # model.macroeconomy.n_mach_RS_avg[t] = mean(cp_id -> model[cp_id].n_mach_ordered_RS, all_cp)

    # Productivity
    # model.macroeconomy.avg_pi_LP[t] = mean(cp_id -> model[cp_id].π_LP, all_cp)
    # model.macroeconomy.avg_pi_EE[t] = mean(cp_id -> model[cp_id].π_EE, all_cp)
    # model.macroeconomy.avg_pi_EF[t] = mean(cp_id -> model[cp_id].π_EF, all_cp)

    # model.macroeconomy.avg_A_LP[t] = mean(kp_id -> model[kp_id].A_LP[end], all_kp)
    # model.macroeconomy.avg_A_EE[t] = mean(kp_id -> model[kp_id].A_EE[end], all_kp)
    # model.macroeconomy.avg_A_EF[t] = mean(kp_id -> model[kp_id].A_EF[end], all_kp)

    # model.macroeconomy.avg_B_LP[t] = mean(kp_id -> model[kp_id].B_LP[end], all_kp)
    # model.macroeconomy.avg_B_EE[t] = mean(kp_id -> model[kp_id].B_EE[end], all_kp)
    # model.macroeconomy.avg_B_EF[t] = mean(kp_id -> model[kp_id].B_EF[end], all_kp)

    # # Production quantity
    # model.macroeconomy.total_Q_cp[t] = sum(cp_id -> model[cp_id].Q[end], all_cp)
    # model.macroeconomy.total_Q_kp[t] = sum(kp_id -> model[kp_id].Q[end], all_kp)
    # if t > 3
    #     total_Q_t = model.macroeconomy.total_Q_cp[t] + model.macroeconomy.total_Q_kp[t]
    #     total_Q_t3 = model.macroeconomy.total_Q_cp[t-3] + model.macroeconomy.total_Q_kp[t-3]
    #     model.macroeconomy.total_Q_growth[t] = (total_Q_t - total_Q_t3) / total_Q_t3
    # end

    # model.macroeconomy.avg_Q_cp[t] = mean(cp_id -> model[cp_id].Q[end], all_cp)
    # model.macroeconomy.avg_Qs_cp[t] = mean(cp_id -> model[cp_id].Qˢ, all_cp)
    # model.macroeconomy.avg_Qe_cp[t] = mean(cp_id -> model[cp_id].Qᵉ, all_cp)
    # model.macroeconomy.avg_Q_kp[t] = mean(kp_id -> model[kp_id].Q[end], all_kp)
    # model.macroeconomy.avg_D_cp[t] = mean(cp_id -> model[cp_id].D[end], all_cp)
    # model.macroeconomy.avg_Du_cp[t] = mean(cp_id -> model[cp_id].Dᵁ[end], all_cp)
    # model.macroeconomy.avg_De_cp[t] = mean(cp_id -> model[cp_id].Dᵉ, all_cp)

    # compute_bankrupties!(bankrupt_cp, bankrupt_kp, model)

    # compute_unsatisfied_demand(model)

    # model.macroeconomy.N_goods[t] = sum(cp_id -> model[cp_id].N_goods, all_cp)
    # model.macroeconomy.avg_N_goods[t] = mean(cp_id -> model[cp_id].N_goods, all_cp)

    # Mean rate of capital utilization
    # model.macroeconomy.cu[t] = mean(cp_id -> model[cp_id].n_machines > 0 ? model[cp_id].cu : 0.5, all_cp)

    # # Average number of machines
    # model.macroeconomy.avg_n_machines_cp[t] = mean(cp_id -> model[cp_id].n_machines, all_cp)

    # Compute GINI coefficients
    # @timeit to "GINI" compute_GINI(all_hh, t, model)

    # compute_I_W_thresholds(all_hh, t, model)
    # compute_α_W_quantiles(all_hh, t, model)
end


"""
Computes GDP based on income of separate sectors, computes partial incomes of sectors
"""
function compute_GDP!(model::ABM)

    # Household income
    model.macroeconomy.GDP_hh[model.t] = sum(hh_id -> model[hh_id].total_I[end], model.all_hh)

    # cp profits
    model.macroeconomy.GDP_cp[model.t] = sum(cp_id -> model[cp_id].Π[end], model.all_cp)
    
    # kp profits
    model.macroeconomy.GDP_kp[model.t] = sum(kp_id -> model[kp_id].Π[end], model.all_kp)

    # ep profits
    Π_ep = model.ep.net_profit[model.t]

    # total nominal GDP
    model.macroeconomy.GDP_nominal[model.t] = (
        model.macroeconomy.GDP_hh[model.t] 
        + model.macroeconomy.GDP_cp[model.t] 
        + model.macroeconomy.GDP_kp[model.t] 
        + Π_ep
    )
    # total real GDP
    model.macroeconomy.GDP_real[model.t] = (model.macroeconomy.GDP_nominal[model.t] 
                                            / model.macroeconomy.p_avg_cp[model.t])

    # Quarterly real GDP growth rates
    t_qrt = 3
    if model.t > t_qrt
        diff_real_GDP = model.macroeconomy.GDP_real[model.t] - model.macroeconomy.GDP_real[model.t-t_qrt]
        model.macroeconomy.GDP_qrt_growth[model.t] = 100 * diff_real_GDP / model.macroeconomy.GDP_real[model.t-t_qrt]
    end

    # Annual real GDP growth rates
    t_ann = 12
    if model.t > t_ann
        diff_real_GDP = model.macroeconomy.GDP_real[model.t] - model.macroeconomy.GDP_real[model.t-t_ann]
        model.macroeconomy.GDP_ann_growth[model.t] = 100 * diff_real_GDP / model.macroeconomy.GDP_real[model.t-t_ann]
    end
end


# function compute_spending!(model::ABM)
#     # Compute planned and actual consumption
#     model.macroeconomy.total_C[model.t] = planned_consumption(model)
#     model.macroeconomy.total_C_actual[model.t] = actual_consumption(model)

#     # Compute total spending on investments
#     model.macroeconomy.total_I[model.t] = total_investments(model)
    
#     # Compute total spending on wages
#     model.macroeconomy.total_w[model.t] = total_wagespending(model)
# end


"""
Computes the ratios of bankrupt bp, lp and kp.
"""
function compute_bankrupties!(bankrupt_cp, bankrupt_kp, model::ABM)
    model.macroeconomy.bankrupt_cp[model.t] = length(bankrupt_cp) / model.i_param.n_cp
    model.macroeconomy.bankrupt_kp[model.t] = length(bankrupt_kp) / model.i_param.n_kp
end


"""
Computes the money supply of households, producers, government 
    and the indexfund
"""
function compute_M!(model::ABM)
    # Wealth of households
    model.macroeconomy.M_hh[model.t] = sum(hh_id -> model[hh_id].W, model.all_hh)

    # Liquid assets of producers
    model.macroeconomy.M_cp[model.t] = sum(cp_id -> model[cp_id].balance.NW, model.all_cp)
    model.macroeconomy.M_kp[model.t] = sum(kp_id -> model[kp_id].balance.NW, model.all_kp)
    model.macroeconomy.M_ep[model.t] = model.ep.NW_ep[model.t]

    # Money owned by government
    model.macroeconomy.M_gov[model.t] = model.gov.MS

    # Money in investment fund
    model.macroeconomy.M_if[model.t] = model.idxf.assets

    # Total amount of money stocks
    model.macroeconomy.M[model.t] = (
        model.macroeconomy.M_hh[model.t] 
        + model.macroeconomy.M_cp[model.t] 
        + model.macroeconomy.M_kp[model.t] 
        + model.macroeconomy.M_ep[model.t] 
        + model.macroeconomy.M_gov[model.t] 
        + model.macroeconomy.M_if[model.t]
    )
end


"""
Computes average wage statistics
"""
function update_wage_stats!(model::ABM)
    model.macroeconomy.w_avg[model.t] = mean(p_id -> model[p_id].w̄[end], model.all_p)
    # model.macroeconomy.w_req_avg[model.t] = mean(hh_id -> model[hh_id].wʳ, model.all_hh)
    # model.macroeconomy.w_sat_avg[model.t] = mean(hh_id -> model[hh_id].wˢ, model.all_hh)
end

# function update_income_stats!(all_hh::Vector{Int}, t::Int, model::ABM)

    # model.macroeconomy.YL_avg[t] = mean(hh_id -> model[hh_id].labor_I, all_hh)

    # model.macroeconomy.YK_avg[t] = mean(hh_id -> model[hh_id].capital_I, all_hh)

    # model.macroeconomy.YUB_avg[t] = mean(hh_id -> model[hh_id].UB_I, all_hh)

    # model.macroeconomy.YSB_avg[t] = mean(hh_id -> model[hh_id].socben_I, all_hh)

    # model.macroeconomy.Y_avg[t] = (model.macroeconomy.YL_avg[t] + model.macroeconomy.YK_avg[t] 
    #                          + model.macroeconomy.YUB_avg[t] + model.macroeconomy.YSB_avg[t])
# end


"""
Updates metrics on aggregate debt levels
"""
function update_debt!(bankrupt_cp, bankrupt_kp, model::ABM)

    model.macroeconomy.debt_cp[model.t] = sum(cp_id -> model[cp_id].balance.debt, model.all_cp)
    model.macroeconomy.debt_kp[model.t] = sum(kp_id -> model[kp_id].balance.debt, model.all_kp)
    model.macroeconomy.debt_tot[model.t] = (
        model.macroeconomy.debt_cp[model.t] 
        + model.macroeconomy.debt_kp[model.t]
        + model.ep.debt_ep[model.t]
    )

    # model.macroeconomy.debt_cp_allowed[model.t] = model.g_param.Λ * sum(cp_id -> model[cp_id].curracc.S, model.all_cp)
    # model.macroeconomy.debt_kp_allowed[model.t] = model.g_param.Λ * sum(kp_id -> model[kp_id].curracc.S, model.all_kp)
    if length(bankrupt_cp) > 0
        model.macroeconomy.debt_unpaid_cp[model.t] = sum(cp_id -> model[cp_id].balance.debt, bankrupt_cp)
    end
    if length(bankrupt_kp) > 0
        model.macroeconomy.debt_unpaid_kp[model.t] = sum(kp_id -> model[kp_id].balance.debt, bankrupt_kp)
    end
end


"""
Computes average price levels and CPI for consumer goods and capital goods.
"""
function compute_price_data!(model::ABM)

    # Compute average price, weighted by market share
    # model.macroeconomy.p_avg_cp[model.t] = avg_cp_price(model)
    model.macroeconomy.p_avg_cp[model.t] = sum(cp_id -> model[cp_id].p[end] * model[cp_id].f[end], model.all_cp)
    # if model.t == 1
    #     model.macroeconomy.CPI_cp[model.t] = 100
    # else
    #     model.macroeconomy.CPI_cp[model.t] = 100 / model.macroeconomy.p_avg_cp[begin] * avg_p_cp_t
    # end

    # Compute average price of capital goods
    # avg_p_kp_t = sum(kp_id -> model[kp_id].p[end] * model[kp_id].f[end], model.all_kp)
    model.macroeconomy.p_avg_kp[model.t] = sum(kp_id -> model[kp_id].p[end] * model[kp_id].f[end], model.all_kp)
    # if model.t == 1
    #     model.macroeconomy.CPI_kp[model.t] = 100
    # else
    #     model.macroeconomy.CPI_kp[model.t] = 100 / model.macroeconomy.p_avg_kp[begin] * avg_p_kp_t
    # end

    # markup_cp(model) = mean(cp_id -> model[cp_id].μ[end], model.all_cp)
    # markup_kp(model) = mean(kp_id -> model[kp_id].μ[end], model.all_kp)
end


# """
# Computes fraction of household that was not satisfied
# """
# function compute_unsatisfied_demand(model::ABM)

#     model.macroeconomy.unsat_demand[model.t] = (sum(cp_id -> model[cp_id].Dᵁ[end], model.all_cp) 
#                                                 / sum(cp_id -> model[cp_id].D[end] + model[cp_id].Dᵁ[end], model.all_cp))

#     model.macroeconomy.unspend_C[model.t] = (1 - sum(hh_id -> model[hh_id].C_actual, model.all_hh) 
#                                             / sum(hh_id -> model[hh_id].C, model.all_hh))

#     model.macroeconomy.unsat_invest[model.t] =  (1 - sum(kp_id -> model[kp_id].Q[end], model.all_kp) 
#                                             / (sum(cp_id -> model.g_param.freq_per_machine * (model[cp_id].n_mach_ordered_EI + model[cp_id].n_mach_ordered_RS), model.all_cp)))

#     model.macroeconomy.unsat_L_demand[model.t] = 1 - model.labormarket.L_hired / model.labormarket.L_demanded
# end


function GINI(x::Vector{Float64}, model::ABM)
    model.ginidata .= abs.(x .- x')
    return sum(model.ginidata) / (2 * model.i_param.n_hh * sum(abs.(x)))
end


"""
GINI approximation function with drastically reduced runtime. Computes using an
    approximation of the surface above and under a Lorenz curve.
"""
function GINI_approx(x::Vector{Float64}, model::ABM)
    model.perc_of_wealth .= (100 * cumsum(sort(x) ./ sum(x)))
    A = sum(model.equal_div .- model.perc_of_wealth)
    B = sum(model.perc_of_wealth)
    G = A / (A + B)
    return G
end


# """
# Computes the GINI coefficient for wealth and income
# """
# function compute_GINI(
#     all_hh::Vector{Int},
#     t::Int,
#     model::ABM;
#     approx_GINI::Bool=true
#     )

#     # Compute GINI for income
#     all_I = map(hh_id -> model[hh_id].total_I[end], all_hh)
#     if approx_GINI
#         model.macroeconomy.GINI_I[t] = GINI_approx(all_I, model)
#     else
#         model.macroeconomy.GINI_I[t] = GINI(all_I, model)
#     end

    # all_I_tmp = zeros(Float64, length(all_I))
    # all_I_absdiff = zeros(Float64, length(all_I))

    # for (i, I1) in enumerate(all_I)
    #     all_I_tmp .= all_I
    #     all_I_tmp .-= I1
    #     all_I_tmp .= abs.(all_I_tmp)
    #     all_I_absdiff[i] = sum(all_I_tmp)
    # end

    # model.macroeconomy.GINI_I[t] = sum(all_I_absdiff) / (2 * length(all_hh)^2 * model.macroeconomy.Y_avg[t])

    # Compute GINI for wealth
    # all_W = map(hh_id -> model[hh_id].W, all_hh)
    # if approx_GINI
    #     model.macroeconomy.GINI_W[t] = GINI_approx(all_W, model)
    # else
    #     model.macroeconomy.GINI_W[t] = GINI(all_W, model)
    # end

    # all_W_tmp = zeros(Float64, length(all_W))
    # all_W_absdiff = zeros(Float64, length(all_W))

    # for (i, W1) in enumerate(all_W)
    #     all_W_tmp .= all_W
    #     all_W_tmp .-= W1
    #     all_W_tmp .= abs.(all_W_tmp)
    #     all_W_absdiff[i] = sum(all_W_tmp)
    # end

    # model.macroeconomy.GINI_W[t] = sum(all_W_absdiff) / (2 * length(all_hh)^2 * model.macroeconomy.M_hh[t] / length(all_hh))
# end


# function compute_I_W_thresholds(
#     all_hh::Vector{Int64},
#     # model.macroeconomy::MacroEconomy,
#     t::Int64,
#     model::ABM
#     )

#     # Establish boundaries of middle 60%
#     start_60 = round(Int64, 0.2 * length(all_hh))
#     end_60 = round(Int64, 0.8 * length(all_hh))

#     # Sort incomes and select income at 20th and 80th percent
#     I_sorted = sort(map(hh_id -> model[hh_id].total_I, all_hh))
#     model.macroeconomy.I_min[t] = I_sorted[begin]
#     model.macroeconomy.I_20[t] = I_sorted[start_60]
#     model.macroeconomy.I_80[t] = I_sorted[end_60]
#     model.macroeconomy.I_max[t] = I_sorted[end]

#     # Sort wealths and select wealth at 20th and 80th percent
#     W_sorted = sort(map(hh_id -> model[hh_id].W, all_hh))
#     model.macroeconomy.W_min[t] = W_sorted[begin]
#     model.macroeconomy.W_20[t] = W_sorted[start_60]
#     model.macroeconomy.W_80[t] = W_sorted[end_60]
#     model.macroeconomy.W_max[t] = W_sorted[end]
# end


# function compute_savings_macro!(model::ABM)
#     if length(model.labormarket.employed_hh) > 1
#         model.macroeconomy.s_emp[model.t] = mean(hh_id -> model[hh_id].s, model.labormarket.employed_hh)
#     end
#     if length(model.labormarket.unemployed_hh) > 1
#         model.macroeconomy.s_unemp[model.t] = mean(hh_id -> model[hh_id].s, model.labormarket.unemployed_hh)
#     end
# end


# function compute_α_W_quantiles(
#     all_hh::Vector{Int64},
#     t::Int64,
#     model::ABM
# )

#     all_α = map(hh_id -> model[hh_id].α, all_hh)
#     model.macroeconomy.α_W_quantiles[:, t] .= Statistics.quantile(all_α, [0.1, 0.25, 0.5, 0.75, 0.9])
# end