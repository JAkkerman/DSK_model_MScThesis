function fire_excess_workers_p!(
    p::Union{ConsumerGoodProducer, CapitalGoodProducer}, 
    model::ABM
    )::Vector{Int}

    if p.ΔLᵈ <= -p.L
        fired_workers = p.employees
    else

        ΔL = -p.ΔLᵈ
        fired_workers = []

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

    # n_to_be_fired = floor(Int, abs(p.ΔLᵈ / 100))

    # if n_to_be_fired >= length(p.employees)
    #     fired_workers = p.employees
    # else
    #     # TODO: find a more sophisticated way to select who is fired
    #     fired_workers = sample(p.employees, n_to_be_fired, replace=false)
    # end

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
function hire_worker_p!(
    p::AbstractAgent, 
    hh::Household
    )

    # update labor stock and desired labor
    push!(p.employees, hh.id)
    p.L += hh.L * hh.skill
    # p.ΔLᵈ -= hh.L * hh.skill
end


function update_L!(
    p::Union{ConsumerGoodProducer, CapitalGoodProducer},
    model::ABM
    )

    p.L = length(p.employees) > 0 ? sum(hh_id -> model[hh_id].L * model[hh_id].skill, p.employees) : 0.0
end


"""
Removes worker from firm
"""
function remove_worker_p!(
    p::AbstractAgent,
    hh::Household
    )

    p.L -= hh.L * hh.skill
    p.employees = filter(hh_id -> hh_id ≠ hh.id, p.employees)
end


"""
Loop over workers and pay out wage.
"""
function pay_workers_p!(
    p::AbstractAgent,
    model::ABM
    )

    total_wage = 0.0
    for hh_id in p.employees
        wage = model[hh_id].w[end] * model[hh_id].L
        total_wage += wage
        receiveincome_hh!(model[hh_id], wage)
    end
    
    p.curracc.TCL += total_wage
end


"""
Updates wage level for producer.
"""
function update_w̄_p!(
    p::AbstractAgent,
    model::ABM
    )

    if length(p.employees) > 0
        shift_and_append!(p.w̄, mean(hh_id -> model[hh_id].w[end], p.employees))
    else
        shift_and_append!(p.w̄, p.w̄[end])
    end
end


"""
Updates market shares of cp firms
"""
function update_marketshare_p!(
    all_p::Vector{Int},
    model::ABM
    )

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
function borrow_funds_p!(
    p::AbstractAgent,
    amount::Float64,
    b::Int64
    )

    # Add debt as repayments for coming periods
    p.debt_installments[2:b+1] .+= amount / b

    # Add received funds as incoming cashflow
    p.curracc.add_debt += amount
    p.balance.debt = sum(p.debt_installments)
end


"""
Adds to-be-repaid amount as an outgoing cashflow to current account.
"""
function payback_debt_p!(
    p::AbstractAgent,
    b::Int64
    )

    # Add repaid debt as outgoing cashflow
    p.curracc.rep_debt = p.debt_installments[1]

    # Shift remaining debt amounts
    p.debt_installments[1:b] .= p.debt_installments[2:b+1]
    p.debt_installments[b+1] = 0.0

    p.balance.debt = sum(p.debt_installments)
end


"""
Checks whether producers are bankrupt and should be replaced, returns vectors
    containing ids of to-be-replaced producers by type
"""
function check_bankrupty_all_p!(
    all_p::Vector{Int},
    all_kp::Vector{Int},
    global_param::GlobalParam, 
    model::ABM
    )::Tuple{Vector{Int}, Vector{Int}, Vector{Int}}

    bankrupt_cp = Vector{Int}()
    bankrupt_kp = Vector{Int}()
    bankrupt_kp_i = Vector{Int}()

    # return bankrupt_bp, bankrupt_lp, bankrupt_kp, bankrupt_kp_i

    # cp_counter = 0
    kp_counter = 0

    for p_id in all_p
        if check_if_bankrupt_p!(model[p_id], global_param.t_wait)
            if typeof(model[p_id]) == ConsumerGoodProducer
                # cp_counter += 1
                push!(bankrupt_cp, p_id)
            else
                kp_counter += 1
                # if kp_counter != length(all_kp)
                if kp_counter < 4
                    push!(bankrupt_kp, p_id)
                    push!(bankrupt_kp_i, model[p_id].kp_i)
                end
            end
        end
    end

    # meanage = length(bankrupt_cp) > 0 ? mean(cp_id -> model[cp_id].age, bankrupt_cp) : nothing
    # meanf = length(bankrupt_cp) > 0 ? mean(cp_id -> model[cp_id].f[end], bankrupt_cp) : nothing
    # meanN = length(bankrupt_cp) > 0 ? mean(cp_id -> model[cp_id].N_goods, bankrupt_cp) : nothing
    # println("mean age: $meanage, meanf: $meanf, meanN: $meanN")

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

    # Remove bankrupt cp ids from households
    for hh_id in all_hh
        remove_bankrupt_producers_hh!(model[hh_id], bankrupt_cp)
    end

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


        # println("   $(typeof(model[p_id])==ConsumerGoodProducer ? "cp" : "kp"), $(model[p_id].L), $(model[p_id].age)")

        # Fire remaining workers
        for hh_id in model[p_id].employees
            set_unemployed_hh!(model[hh_id])
        end

        # TODO: TEMP SOLUTION, DESCRIBE IT WORKS
        total_unpaid_net_debt += (model[p_id].balance.debt - model[p_id].balance.NW)

        # Remove firm agents from model
        kill_agent!(p_id, model)
    end

    deduct_unpaid_net_debts_if!(indexfund_struct, total_unpaid_net_debt)
end


"""
Checks if producers must be declared bankrupt
"""
function check_if_bankrupt_p!(
    p::AbstractAgent,
    t_wait::Int
    )::Bool

    # return false

    if (typeof(p) == ConsumerGoodProducer && p.age > t_wait 
        && (p.f[end] <= 0.0001 || p.balance.EQ < 0))
        # println("           $(p.f[end]), $(p.ΔLᵈ), $(p.L)")
    # if p.age > t_wait && p.f[end] <= 0.0001
    # if p.age > t_wait && p.balance.EQ < 0
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
    global_param::GlobalParam;
    is_EF::Bool = false
    )::Float64

    κ_i = rand(Beta(global_param.α1, global_param.β1))
    κ_i = global_param.κ_lower + κ_i * (global_param.κ_upper - global_param.κ_lower)

    return is_EF ? TP_last*(1 - κ_i) : TP_last*(1 + κ_i)
end


"""
Updates the mean skill level of employees.
"""
function update_mean_skill_p!(
    p::Union{ConsumerGoodProducer, CapitalGoodProducer},
    model::ABM
    )

    p.mean_skill = length(p.employees) > 0 ? mean(hh_id -> model[hh_id].skill, p.employees) : 0.0
end