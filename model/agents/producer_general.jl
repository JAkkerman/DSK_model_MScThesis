"""Abstract producer struct
"""
abstract type Producer <: AbstractAgent end


"""
Computes the cost of production
"""
function cop(w::Float64, π_LP::Float64, τᴱ::Float64, pₑ::Float64,
             π_EE::Float64, τᶜ::Float64, π_EF::Float64)
    return w / π_LP + (τᴱ + pₑ) / π_EE + τᶜ * π_EF
end


"""
Computes the cost of production
"""
function cop(w::Float64, τᴱ::Float64, pₑ::Float64, τᶜ, brochure::Brochure)
    return cop(w, brochure.A_LP, τᴱ, pₑ, brochure.A_EE, τᶜ, brochure.A_EF)
end


check_req_wage(hh_id, p_id, model) = model[hh_id].wʳ / model[hh_id].skill <= model[p_id].wᴼ_max
check_lab_supply(hh_id, demanded_labor, allowed_excess_L, model) = model[hh_id].L * model[hh_id].skill <= demanded_labor + allowed_excess_L


"""
Lets producers select employees to fire.
"""
function fire_excess_workers_p!(p::Producer, model::ABM)::Vector{Int64}

    if p.ΔLᵈ <= -p.L
        # Fire all employees if desired decrease in labor exceeds labor stock.
        fired_workers = p.employees
    else

        ΔL = -p.ΔLᵈ
        fired_workers = Int64[]

        # Sort workers based on their relative expensiveness
        sort!(p.employees, by = hh_id -> model[hh_id].w[end] / model[hh_id].skill, rev=true)

        for hh_id in p.employees
            if ΔL < 50
                break
            end
            push!(fired_workers, hh_id)
            ΔL -= (model[hh_id].L * model[hh_id].skill)
        end

    end

    # Remove employees from labor stock
    if length(fired_workers) > 0

        for hh_id in fired_workers
            remove_worker_p!(p, model[hh_id])
        end

        return fired_workers
    else
        return Vector{Int}()
    end

end


"""
Hires workers, adds worker id to worker array, updates labor stock.
"""
function hire_worker_p!(p::Producer, hh::Household)
    push!(p.employees, hh.id)
    p.L += hh.L * hh.skill
end


"""
Update amount of labor units in firm
"""
function update_L!(p::Producer, model::ABM)
    p.L = length(p.employees) > 0 ? sum(hh_id -> model[hh_id].L * model[hh_id].skill, p.employees) : 0.0
end


"""
Removes worker from firm
"""
function remove_worker_p!(p::Producer, hh::Household)
    p.L -= hh.L * hh.skill
    p.employees = filter(hh_id -> hh_id ≠ hh.id, p.employees)
end


"""
Loop over workers and pay out wage.
"""
function pay_workers_p!(p::Producer, model::ABM)

    total_wage = 0.0
    total_incometax = 0.0

    for hh_id in p.employees
        wage = model[hh_id].w[end] * model[hh_id].L
        total_wage += wage
        total_incometax += model.gov.τᴵ * wage
        receiveincome_hh!(model[hh_id], wage * (1 - model.gov.τᴵ))
    end
    
    receive_incometax_gov!(model.gov, total_incometax, model.t)
    p.curracc.TCL = total_wage
end


"""
Updates wage level for producer.
"""
function update_w̄_p!(p::Producer, model::ABM)

    if length(p.employees) > 0
        shift_and_append!(p.w̄, mean(hh_id -> model[hh_id].w[end], p.employees))
    else
        shift_and_append!(p.w̄, p.w̄[end])
    end
end


"""
Updates market shares of cp firms.
"""
function update_marketshare_p!(all_p::Vector{Int}, model::ABM)

    # Compute total market size of bp and lp markets
    market = sum(p_id -> model[p_id].D[end], all_p)

    for p_id in all_p
        if market == 0
            f = 1 / length(all_p)
        else
            f = model[p_id].D[end] / market
        end
        shift_and_append!(model[p_id].f, f)
    end
end


"""
Adds borrowed amount as an incoming cashflow to current account.
"""
function borrow_funds_p!(p::Producer, amount::Float64, b::Int64)

    # Add debt as repayments for coming periods
    p.debt_installments[2:b+1] .+= amount / b

    # Add received funds as incoming cashflow
    p.curracc.add_debt += amount
    p.balance.debt = sum(p.debt_installments)
end


"""
Adds to-be-repaid amount as an outgoing cashflow to current account.
"""
function monthly_debt_payback_p!(p::Producer, b::Int64)

    # Add repaid debt as outgoing cashflow
    p.curracc.rep_debt = p.debt_installments[begin]

    # Shift remaining debt amounts
    p.debt_installments[begin:b] .= p.debt_installments[begin+1:b+1]
    p.debt_installments[b+1] = 0.

    p.balance.debt = sum(p.debt_installments)
end


"""
Pays back additional amount of debt.
"""
function singular_debt_payback_p!(p::Producer, excess_NW::Float64)

    payoff_debt = min(p.balance.debt, excess_NW)
    excess_NW_after_debt = excess_NW - payoff_debt

    can_pay_full_debt = payoff_debt == p.balance.debt

    if can_pay_full_debt
        p.curracc.rep_debt = payoff_debt
        p.debt_installments .= 0.
        p.balance.debt = 0.
    else
        i = 1
        while p.debt_installments[end - i] < excess_NW
            excess_NW -= p.debt_installments[end - i]
            p.debt_installments[end - i] = 0.
            i += 1
        end
        p.debt_installments[end - i] -= excess_NW
        p.balance.debt = sum(p.debt_installments)
        p.curracc.rep_debt += payoff_debt
    end

    return excess_NW_after_debt
end


"""
Checks whether producers are bankrupt and should be replaced, returns vectors
    containing ids of to-be-replaced producers by type
"""
function check_bankrupty_all_p!(
    all_p::Vector{Int},
    all_kp::Vector{Int},
    globalparam::GlobalParam, 
    model::ABM
    )::Tuple{Vector{Int}, Vector{Int}, Vector{Int}}

    bankrupt_cp = Int64[]
    bankrupt_kp = Int64[]
    bankrupt_kp_i = Int64[]

    cp_counter = 0
    kp_counter = 0

    n_cp = length(all_p) - length(all_kp)

    for p_id in all_p
        if check_if_bankrupt_p!(model[p_id], globalparam.t_wait)
            if (typeof(model[p_id]) == ConsumerGoodProducer 
                && cp_counter < n_cp - 1 
                && length(model[p_id].Ξ) > 0)
                push!(bankrupt_cp, p_id)
                cp_counter += 1
            elseif (typeof(model[p_id]) == CapitalGoodProducer
                && kp_counter < length(all_kp) - 1)
                push!(bankrupt_kp, p_id)
                push!(bankrupt_kp_i, model[p_id].kp_i)
                kp_counter += 1
            end
        end
    end

    return bankrupt_cp, bankrupt_kp, bankrupt_kp_i
end


"""
Kills all bankrupt producers.
    - for cp, removes bankrupt companies from households.
    - for all, fires all remaining workers.
    - for all, removes firm agent from model.
"""
function kill_all_bankrupt_p!(
    bankrupt_cp::Vector{Int},
    bankrupt_kp::Vector{Int},
    all_hh::Vector{Int},
    all_kp::Vector{Int},
    labormarket_struct,
    indexfund_struct,
    model::ABM
    )

    # Remove bankrupt cp ids from kp historical clients
    for kp_id in all_kp
        remove_bankrupt_HC_kp!(model[kp_id], bankrupt_cp)
    end

    # All employees are declared unemployed
    all_employees = Vector{Int}()
    for p_id in Iterators.flatten((bankrupt_cp, bankrupt_kp))
        append!(all_employees, model[p_id].employees)
    end
    
    update_firedworker_lm!(labormarket_struct, all_employees)

    # Fire all workers still remaining in the firm, remove firm
    total_unpaid_net_debt = 0.0
    for p_id in Iterators.flatten((bankrupt_cp, bankrupt_kp))

        # Fire remaining workers
        for hh_id in model[p_id].employees
            set_unemployed_hh!(model[hh_id])
        end

        total_unpaid_net_debt += (model[p_id].balance.debt - model[p_id].balance.NW)

        # Remove firm agents from model
        kill_agent!(p_id, model)
    end

    deduct_unpaid_net_debts_if!(indexfund_struct, total_unpaid_net_debt)
end


"""
Checks if producers must be declared bankrupt
"""
function check_if_bankrupt_p!(p::Producer, t_wait::Int64)::Bool

    if (typeof(p) == ConsumerGoodProducer && p.age > t_wait 
        && (p.f[end] <= 0.0001 || p.balance.EQ < 0))
        return true
    elseif (typeof(p) == CapitalGoodProducer && p.age > t_wait 
            &&  p.balance.EQ < 0)
        return true
    end
    return false
end 


"""
Determines new productivity level, resulting from innovation process for both kp and ep.
"""
function update_techparam_p(
    TP_last::Float64, 
    globalparam::GlobalParam;
    is_EF::Bool = false
    )::Float64

    κ_i = rand(Beta(globalparam.α1, globalparam.β1))
    κ_i = globalparam.κ_lower + κ_i * (globalparam.κ_upper - globalparam.κ_lower)

    return is_EF ? TP_last * (1 - κ_i) : TP_last * (1 + κ_i)
end


"""
Updates the mean skill level of employees.
"""
function update_mean_skill_p!(p::Producer, model::ABM)
    p.mean_skill = length(p.employees) > 0 ? mean(hh_id -> model[hh_id].skill, p.employees) : 0.0
end


function update_age!(p::Producer)
    p.age += 1
end