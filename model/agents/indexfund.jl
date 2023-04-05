Base.@kwdef mutable struct IndexFund
    # T::Int=T 
    assets::Float64 = 0.                            # Total amount of assets alocated in the fund
    funds_inv::Vector{Float64} = zeros(Float64, T)  # Amount of funds used for investment in last period
    returns_investments::Float64 = 0.               # Return rates of investments over last period
end


"""
Initializes index fund struct
"""
function initialize_indexfund(T)::IndexFund

    indexfund = IndexFund(
        funds_inv = zeros(Float64, T)
    )
    
    return indexfund
end


"""
Takes dividends from producers
"""
function receive_dividends_if!(indexfund::IndexFund, dividends::Float64)
    indexfund.assets += dividends
end


"""
Deducts funds for investment in company
"""
function decide_investments_if!(
    indexfund::IndexFund,
    all_req_NW::Float64,
    t::Int
    )::Float64

    frac_NW_if = indexfund.assets > 0 ? min(indexfund.assets / all_req_NW, 1.0) : 0.0
    indexfund.funds_inv[t] = all_req_NW * frac_NW_if
    indexfund.assets -= indexfund.funds_inv[t]
    return frac_NW_if
end


"""
Deducts funds for net debts lost
"""
function deduct_unpaid_net_debts_if!(indexfund::IndexFund, total_unpaid_net_debt::Float64)
    indexfund.assets -= total_unpaid_net_debt
end


"""
Distributes dividends over participants in indexfund
"""
function distribute_dividends_if!(
    indexfund::IndexFund,
    government::Government,
    all_hh::Vector{Int64},
    τᴷ::Float64,
    t::Int64,
    model::ABM
    )

    # Distribute proportional to wealth
    all_W = map(hh_id -> model[hh_id].W, model.all_hh)
    max_W = maximum(all_W)
    total_W = sum(all_W)

    # Compute return rates:
    total_dividends = (1 - τᴷ) * indexfund.assets
    indexfund.returns_investments = total_dividends / total_W

    for hh_id in model.all_hh
        # Do not award to most wealthy household to avoid one household
        # taking over all wealth
        dividend_share = 0.
        if t < 10 
            dividend_share = (model[hh_id].W / total_W)
        elseif model[hh_id].W ≠ max_W
            dividend_share = (model[hh_id].W / (total_W - max_W))
        end

        dividend = dividend_share * total_dividends
        receiveincome_hh!(model[hh_id], dividend; capgains=true)
    end

    # Pay capgains tax, reset assets back to zero
    total_capgains_tax = τᴷ * indexfund.assets
    receive_capgains_tax_gov!(government, total_capgains_tax, model.t)
    indexfund.assets = 0.
end


"""
Lets government issue bonds on the capital market.
"""
function issuegovbonds(indexfund::IndexFund, govdeficit::Float64)
    indexfund.assets -= govdeficit
end