@with_kw mutable struct EnergyProducer
    T::Int64=T                                    # Total number of iterations

    Dₑ::Vector{Float64} = zeros(Float64, T)     # Demand for energy units over time
    Q̄ₑ::Vector{Float64} = zeros(Float64, T)     # Maximum production of units over time

    # Prices, cost and investments
    μₑ::Float64                                 # Markup to determine price
    Πₑ::Vector{Float64} = zeros(Float64, T)     # Profits over time
    NWₑ::Vector{Float64} = zeros(Float64, T)    # Stock of liquid assets over time
    pₑ::Vector{Float64} = zeros(Float64, T)     # Price of energy over time
    PCₑ::Vector{Float64} = zeros(Float64, T)    # Cost of generating Dₑ(t) units of energy over time
    FU::Vector{Float64} = zeros(Float64, T)     # Number of fuel units used for production
    ICₑ::Vector{Float64} = zeros(Float64, T)    # Expansion and replacement investments over time
    RDₑ::Vector{Float64} = zeros(Float64, T)    # R&D expenditure over time
    IN_g::Vector{Float64} = zeros(Float64, T)   # R&D spending allocated to innovation in green tech
    IN_d::Vector{Float64} = zeros(Float64, T)   # R&D spending allocated to innovation in dirty tech
    EIᵈ::Vector{Float64} = zeros(Float64, T)    # Desired amount of units of expansionary investments
    RSᵈ::Vector{Float64} = zeros(Float64, T)    # Desired amount of units of replacement investments
    ECₑ::Vector{Float64} = zeros(Float64, T)    # Cost of spansionary investment
    carbontax::Vector{Float64} = zeros(Float64, T) # Paid carbon taxes

    # Technological parameters
    IC_g::Vector{Float64}                       # Fixed investment cost of the cheapest new green power plant
    Aᵀ_d::Vector{Float64} = zeros(Float64, T)   # Thermal efficiency of new power plants
    emᵀ_d::Vector{Float64} = zeros(Float64, T)  # Emissions of new power plants
    c_d::Vector{Float64} = zeros(Float64, T)    # Discounted production cost of the cheapest dirty plant 

    # Owned power plants
    green_portfolio::Vector{PowerPlant}         # Portfolio of all green powerplants
    green_capacity::Vector{Float64} = zeros(Float64, T) # capacity of set of green powerplants
    dirty_portfolio::Vector{PowerPlant}         # Portfolio of all dirty powerplants
    dirty_capacity::Vector{Float64} = zeros(Float64, T) # Capacity of set of dirty powerplants,
    green_frac_prod::Vector{Float64} = zeros(Float64, T) # Green fraction of total production
    infra_marg::Vector{PowerPlant} = []         # Infra-marginal plants
    pp_tb_repl::Vector{PowerPlant} = []         # Power plants to be replaced

    emissions::Vector{Float64} = zeros(Float64, T)
end


function initialize_energy_producer(
    T::Int64,
    initparam::InitParam,
    τᶜ::Float64,
    globalparam::GlobalParam
    )::EnergyProducer

    # Initialize power plants
    n_pp = ceil(Int64, initparam.n_powerplants_init / globalparam.freq_per_powerplant)
    n_pp_green = ceil(Int64, initparam.frac_green * n_pp)

    # Initialize green power plants
    green_portfolio = []
    for _ in 1:n_pp_green
        green_pp = PowerPlant(
                    type = "Green",
                    age = sample(0:globalparam.ηₑ),
                    c = 0.0,
                    freq = globalparam.freq_per_powerplant,
                    capacity = globalparam.freq_per_powerplant,
                    Aᵀ = 0.0,
                    em = 0.0
                   )
        push!(green_portfolio, green_pp)
    end

    # Initialize dirty power plants
    dirty_portfolio = []
    for _ in n_pp_green+1:n_pp
        dirty_pp = init_powerplant(
                    "Dirty",
                    sample(0:globalparam.ηₑ),
                    0.0,
                    initparam.Aᵀ_0,
                    initparam.emᵀ_0,
                    globalparam
                   )
        update_c_pp!(dirty_pp, globalparam.p_f, τᶜ)
        push!(dirty_portfolio, dirty_pp)
    end

    # Initialize ep struct
    energy_producer = EnergyProducer(
                        T = T,
                        green_portfolio = green_portfolio,
                        dirty_portfolio = dirty_portfolio,
                        μₑ = initparam.μₑ,
                        Aᵀ_d = fill(initparam.Aᵀ_0, T),
                        emᵀ_d = fill(initparam.emᵀ_0, T),
                        IC_g = fill(initparam.IC_g_0, T)
                      )
    return energy_producer
end



"""
PRODUCTION
"""

"""
Production process of ep
"""
function produce_energy_ep!(
    ep::EnergyProducer,
    government::Government,
    all_cp::Vector{Int64},
    all_kp::Vector{Int64},
    globalparam::GlobalParam,
    indexfund::IndexFund,
    frac_green::Float64,
    t::Int64,
    t_warmup::Int64,
    model::ABM
    )

    # Update age for all pp and cost figure for dirty pp
    for pp ∈ Iterators.flatten((ep.green_portfolio, ep.dirty_portfolio))

        update_age_pp!(pp)

        if pp ∈ ep.dirty_portfolio
            update_c_pp!(pp, globalparam.p_f, government.τᶜ)
        end
    end

    # Update production capacities
    update_capacities_ep!(ep, t)
    compute_Q̄_ep!(ep, t)
    
    # Update cost frontier of ep
    compute_c_ep!(ep, government, globalparam.p_f, t)

    # Determine demand for energy
    compute_Dₑ_ep!(ep, all_cp, all_kp, t, model)

    # Check if production capacity needs to be expanded and old pp replaced
    plan_investments_ep!(ep, government, globalparam, frac_green, t, t_warmup)

    # Choose pp to use in production
    choose_powerplants_ep!(ep, t)
    compute_FU_ICₑ_ep!(ep, globalparam.p_f, t)
    compute_emissions_ep!(ep, t)

    pay_carbontax_ep!(ep, government, t)

    # Compute profits
    compute_Πₑ_NWₑ_ep!(ep, t)
    pay_dividends_ep!(ep, indexfund, t)
end


"""
Lets ep choose power plants to produce energy demand with
"""
function choose_powerplants_ep!(
    ep::EnergyProducer,
    t::Int64
    )

    # Check if all production can be done using green tech, if not, compute cost
    # of production using dirty tech
    if ep.Dₑ[t] < ep.green_capacity[t]
        ep.infra_marg = copy(ep.green_portfolio)
        ep.green_frac_prod[t] = 1.0
    else

        ep.infra_marg = copy(ep.green_portfolio)
        total_capacity = ep.green_capacity[t]

        # Sort dirty portfolio as to take low cost power plants first
        sort!(ep.dirty_portfolio, by = pp -> pp.c)

        for dirty_pp in ep.dirty_portfolio
            push!(ep.infra_marg, dirty_pp)
            total_capacity += dirty_pp.capacity * dirty_pp.Aᵀ
            if total_capacity >= ep.Dₑ[t]
                break
            end
        end

        ep.green_frac_prod[t] = length(ep.green_portfolio) / length(ep.infra_marg)
    end
end


function pay_dividends_ep!(
    ep::EnergyProducer, 
    indexfund::IndexFund, 
    t::Int64
    )

    # TODO: describe

    b = 3

    # ep should have at least enough NW to pay cost of production for b months plus an
    # investment in a green plant
    req_NW = b * ep.PCₑ[t] + ep.IC_g[t]

    # Pay expenses to if, as 'indeterminate' producer receives these expenses,
    # plus all excess liquid assets
    dividends = ep.PCₑ[t] + ep.ICₑ[t] + ep.RDₑ[t] + max(ep.NWₑ[t] - req_NW, 0)
    ep.NWₑ[t] = min(ep.NWₑ[t], req_NW)

    receive_dividends_if!(indexfund, dividends)
end


"""
INVESTMENT
"""


"""
Investment process of ep
"""
function plan_investments_ep!(
    ep::EnergyProducer,
    government::Government,
    globalparam::GlobalParam,
    frac_green::Float64, 
    t::Int64,
    t_warmup::Int64
    )

    compute_RSᵈ_ep!(ep, globalparam.ηₑ, t)

    compute_EIᵈ_ep!(ep, t)

    expand_and_replace_pp_ep!(ep, globalparam, frac_green, t, t_warmup)
end


"""

"""
function expand_and_replace_pp_ep!(
    ep::EnergyProducer,
    globalparam::GlobalParam,
    frac_green::Float64,
    t::Int64,
    t_warmup::Int64
    )

    n_add_pp = ceil(Int64, (ep.EIᵈ[t] + ep.RSᵈ[t]) / globalparam.freq_per_powerplant)

    # Determine what share of additional powerplants should be green and dirty
    if t >= t_warmup
        # If after warmup period, ep can plan investment based on cheapest tech
        n_add_green_pp = 0
        n_add_dirty_pp = 0
        if ep.IC_g[t] <= globalparam.bₑ * ep.c_d[t]
            n_add_green_pp = n_add_pp
        else
            n_add_dirty_pp = n_add_pp
        end
    else
        # If in warmup period, ep will replace such that share of green tech remains
        # at init level

        n_add_green_pp = max(ceil(frac_green * (ep.green_capacity[t] + ep.dirty_capacity[t]
                              + n_add_pp) - ep.green_capacity[t]), 0)
        n_add_dirty_pp = max(n_add_pp - n_add_green_pp, 0)
    end

    # TODO: if expanding green, invested sum should go somewhere!

    # Add green power plants
    if n_add_green_pp > 0
        
        # Invest green
        ep.ECₑ[t] = ep.IC_g[t] * ep.EIᵈ[t]
        
        # Build new green pp
        for _ in 1:n_add_green_pp
            green_pp = PowerPlant(
                type = "Green",
                age = 0,
                c = 0.0,
                freq = globalparam.freq_per_powerplant,
                capacity = globalparam.freq_per_powerplant,
                Aᵀ = 0.0,
                em = 0.0
            )
            push!(ep.green_portfolio, green_pp)
        end
    end

    # Add dirty power plants
    if n_add_dirty_pp > 0

        # Invest dirty
        for _ in 1:n_add_dirty_pp
            dirty_pp = init_powerplant(
                            "Dirty",
                            0,
                            ep.c_d[t],
                            ep.Aᵀ_d[t],
                            ep.emᵀ_d[t],
                            globalparam
                       )
            push!(ep.dirty_portfolio, dirty_pp)
        end
    end

    # Discard old machines
    filter!(pp -> pp ∉ ep.pp_tb_repl, ep.green_portfolio)
    filter!(pp -> pp ∉ ep.pp_tb_repl, ep.dirty_portfolio)
end


"""
INNOVATION
"""

"""
Innocation process
"""
function innovate_ep!(
    ep::EnergyProducer,
    globalparam::GlobalParam,
    t::Int64
    )

    # Compute R&D spending (Lamperti et al (2018), eq 18)
    ep.RDₑ[t] = t > 1 ? globalparam.νₑ * ep.pₑ[t-1] * ep.Dₑ[t-1] : 0.0

    # Compute portions of R&D spending going to innovation in green and dirty tech
    #   (Lamperti et al (2018), eq 18.5)
    ep.IN_g[t] = globalparam.ξₑ * ep.RDₑ[t]
    ep.IN_d[t] = (1 - globalparam.ξₑ) * ep.RDₑ[t]

    # Define success probabilities of tech search (Lamperti et al (2018), eq 19)
    θ_g = 1 - exp(-globalparam.ζ_ge * ep.IN_g[t])
    θ_d = 1 - exp(-globalparam.ζ_de * ep.IN_d[t])

    # Candidate innovation for green tech
    if rand(Bernoulli(θ_g))
        # Draw from Beta distribution
        κ_g = rand(Beta(globalparam.α1, globalparam.β1))

        # Scale over supports
        κ_g = globalparam.κ_lower + κ_g * (globalparam.κ_upper - globalparam.κ_lower)

        # Compute possible cost, replace all future if better
        #   (Lamperti et al (2018), eq 20)
        poss_IC_g = ep.IC_g[t] * (1 - κ_g)
        ep.IC_g[t:end] .= poss_IC_g < ep.IC_g[t] ? poss_IC_g : ep.IC_g[t]
    end

    # Candidate innovation for dirty tech
    if rand(Bernoulli(θ_d))
        # Draw from Beta distribution
        κ_d_A, κ_d_em = rand(Beta(globalparam.α1, globalparam.β1), 2)

        # Scale over supports
        κ_d_A = globalparam.κ_lower + κ_d_A * (globalparam.κ_upper - globalparam.κ_lower)
        κ_d_em = globalparam.κ_lower + κ_d_em * (globalparam.κ_upper - globalparam.κ_lower)

        # Compute possible thermal efficiency, replace all future if better
        #   (Lamperti et al (2018), eq 21)
        poss_A_d = ep.Aᵀ_d[t] * (1 + κ_d_A)
        ep.Aᵀ_d[t:end] .= poss_A_d > ep.Aᵀ_d[t] ? poss_A_d : ep.Aᵀ_d[t]

        # Compute possible emissions, replace all future if better
        #   (Lamperti et al (2018), eq 21)
        poss_em_d = ep.emᵀ_d[t] * (1 - κ_d_em)
        ep.emᵀ_d[t:end] .= poss_em_d < ep.emᵀ_d[t] ? poss_em_d : ep.emᵀ_d[t]
    end
end



"""
COMPUTE AND UPDATE FUNCTIONS
"""


"""
Computes the profits and new liquid assets resulting from last periods production.
    Lamperti et al (2018), eq 10.
"""
function compute_Πₑ_NWₑ_ep!(
    ep::EnergyProducer, 
    t::Int64
    )

    ep.Πₑ[t] = ep.pₑ[t] * ep.Dₑ[t] - ep.PCₑ[t] - ep.ICₑ[t] - ep.RDₑ[t] - ep.carbontax[t]
    ep.NWₑ[t] = t > 1 ? ep.NWₑ[t-1] + ep.Πₑ[t] : ep.Πₑ[t]
end


"""
Computes cost of production PCₑ of infra marginal power plants
    Lamperti et al (2018), eq 12.
"""
function compute_PCₑ_ep!(
    ep::EnergyProducer,
    t::Int64
    )

    dirty_cost = 0.0

    for pp in ep.infra_marg
        if pp ∈ ep.dirty_portfolio
            dirty_cost += pp.freq * pp.c * pp.Aᵗ
        end
    end

    ep.PCₑ[t] = dirty_cost
end


"""
Computes the price for each sold energy unit
"""
function compute_pₑ_ep!(
    ep::EnergyProducer, 
    t::Int64
    )

    # c̄ = length(ep.dirty_portfolio) > 0 ? ep.dirty_portfolio[end].c : 0.0
    # ep.pₑ[t] = t > 1 && ep.Dₑ[t-1] <= ep.green_capacity[t] ? ep.μₑ : c̄ + ep.μₑ

    if length(ep.dirty_portfolio) > 0
        c̄ = maximum(pp -> pp.c, ep.dirty_portfolio)
    else
        c̄ = 0.
    end

    ep.pₑ[t] = t > 1 ? ep.μₑ + (1 - ep.green_frac_prod[t-1]) * c̄ : ep.μₑ
end


"""
Updates capacity figures for green and dirty technologies
    Lamperti et al (2018) eq 14.
"""
function update_capacities_ep!(
    ep::EnergyProducer,
    t::Int64
    )

    ep.green_capacity[t] = length(ep.green_portfolio) > 0 ? sum(pp -> pp.freq, ep.green_portfolio) : 0.0
    ep.dirty_capacity[t] = length(ep.dirty_portfolio) > 0 ? sum(pp -> pp.freq, ep.dirty_portfolio) : 0.0
end


"""
Computes the maximum production level Q̄
    Lamperti et al (2018) eq 15.
"""
function compute_Q̄_ep!(
    ep::EnergyProducer, 
    t::Int64
    )

    ep.Q̄ₑ[t] = ep.green_capacity[t] 

    if length(ep.dirty_portfolio) > 0
        ep.Q̄ₑ[t] += length(ep.dirty_portfolio) > 0 ? sum(pp -> pp.freq * pp.Aᵀ, ep.dirty_portfolio) : 0.0
    end
end


"""
Computes the desired expansion investment
    Lamperti et al (2018) eq 16.
"""
function compute_EIᵈ_ep!(
    ep::EnergyProducer,
    t::Int64
    )

    # TODO: I dont know yet what Kᵈ is, likely has to be changed
    ep.EIᵈ[t] = ep.Q̄ₑ[t] < ep.Dₑ[t] ? ep.Dₑ[t] - (ep.green_capacity[t] + ep.dirty_capacity[t]) : 0.0
end


"""
Computes the desired replacement investment

"""
function compute_RSᵈ_ep!(
    ep::EnergyProducer,
    ηₑ::Int64,
    t::Int64
    )

    # Select powerplants to be replaced
    ep.pp_tb_repl = []
    for pp in Iterators.flatten((ep.green_portfolio, ep.dirty_portfolio))
        if pp.age >= ηₑ
            push!(ep.pp_tb_repl, pp)
        end
    end
    ep.RSᵈ[t] = length(ep.pp_tb_repl) > 0 ? sum(pp -> pp.freq, ep.pp_tb_repl) : 0.0
end


"""
Computes total energy demand in period
"""
function compute_Dₑ_ep!(
    ep::EnergyProducer,
    all_cp::Vector{Int64},
    all_kp::Vector{Int64},
    t::Int64,
    model::ABM
    )

    ep.Dₑ[t] = sum(cp_id -> model[cp_id].EU, all_cp) + sum(kp_id -> model[kp_id].EU, all_kp)
end


"""
Computes cost of dirty powerplant at technological frontier
"""
function compute_c_ep!(
    ep::EnergyProducer,
    government::Government, 
    p_f::Float64, 
    t::Int64
    )

    ep.c_d[t] = p_f / ep.Aᵀ_d[t] + government.τᶜ * ep.emᵀ_d[t]
end


"""
Updates age of power plants
"""
function update_age_pp_ep!(
    ep::EnergyProducer
    )

    for pp in Iterators.flatten((ep.green_portfolio, ep.dirty_portfolio))
        pp.age += 1
    end
end

"""
Computes emissions following from energy production
"""
function compute_emissions_ep!(
    ep::EnergyProducer, 
    t::Int64
    )

    # Emissions remain at default value of zero if only green pp used,
    # Otherwise compute emissions of dirty pp in infra-marginal stock
    if ep.Dₑ[t] > ep.green_capacity[t]

        req_capacity = ep.Dₑ[t] - ep.green_capacity[t]
        total_emissions = 0.0

        # Only use fraction of machines required, in order as sorted for costs
        for pp in ep.infra_marg
            if pp ∈ ep.dirty_portfolio
                total_emissions += max(min(req_capacity / pp.capacity, 1.0), 0) * pp.capacity * pp.em
                req_capacity -= pp.capacity
            end
        end

        ep.emissions[t] = total_emissions
    end
end


function pay_carbontax_ep!(
    ep::EnergyProducer,
    government::Government,
    t::Int64
    )

    ep.carbontax[t] = government.τᶜ * ep.emissions[t]
    government.curracc.Rev_τᶜ[t] += ep.carbontax[t]
end


function compute_FU_ICₑ_ep!(
    ep::EnergyProducer,
    p_f::Float64, 
    t::Int64
    )

    ep.FU[t] = length(ep.infra_marg) > 0 ? sum(pp-> pp ∈ ep.dirty_portfolio ? pp.capacity / pp.Aᵀ : 0.0, ep.infra_marg) : 0.0
    ep.ICₑ[t] = p_f * ep.FU[t]
end