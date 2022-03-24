mutable struct CapitalGoodProducer <: AbstractAgent
    id :: Int                               # global id
    kp_i :: Int                             # kp id, used for distance matrix
    first_gen :: Bool                       # shows if producer is in first generation 
    A :: Vector{Float64}                    # labor prod sold product
    B :: Vector{Float64}                    # labor prod own production
    p :: Vector{Float64}                    # hist price data
    c :: Vector{Float64}                    # hist cost data
    μ :: Vector{Float64}                    # hist markup rate
    employees :: Vector{Int}                # employees in company
    L :: Float64                            # labor units in company
    ΔLᵈ :: Float64                          # desired change in labor force
    w̄ :: Vector{Float64}                    # wage level
    wᴼ :: Float64                           # offered wage
    wᴼ_max :: Float64                       # maximum offered wage
    O :: Float64                            # total amount of machines ordered
    prod_queue :: Array                     # production queue of machines
    Q :: Vector{Float64}                    # production quantities
    RD :: Vector{Float64}                   # hist R&D expenditure
    IM :: Vector{Float64}                   # hist immitation expenditure
    IN :: Vector{Float64}                   # hist innovation expenditure
    D :: Vector{Float64}                    # hist demand
    HC :: Vector{Int}                       # hist clients
    Π :: Vector{Float64}                    # hist profits
    debt_installments :: Vector{Float64}    # installments of debt repayments
    f :: Vector{Float64}                    # market share
    brochure                                # brochure
    orders :: Array                         # orders
    balance :: Balance                      # balance sheet
    curracc :: FirmCurrentAccount           # current account
end


"""
Initializes kp agent, default is the heterogeneous state, otherwise properties are given
    as optional arguments.
"""
function initialize_kp(
    id::Int, 
    kp_i::Int,
    n_captlgood::Int;
    NW=1000,
    A=1.0,
    B=1.0,
    p=1.0,
    μ1=0.2,
    w̄=1.0,
    wᴼ=1.0,
    Q=100,
    D=100,
    f=1/n_captlgood,
    first_gen=true
    )

    balance = Balance(               
        0.0,                    # - N: inventory
        0.0,                    # - K: capital
        NW,                     # - NW: liquid assets
        0.0,                    # - debt: debt
        NW                      # - EQ: equity
    )
    
    curracc = FirmCurrentAccount(0,0,0,0,0,0,0)

    kp = CapitalGoodProducer(   # initial parameters based on rer98
        id,                     # global id
        kp_i,                   # kp_i, used for distance matrix
        first_gen,              # shows if producer is in first generation
        [A],                    # A: labor prod sold product
        [B],                    # B: labor prod own production
        [p],                    # p: hist price data
        [],                     # c: hist cost data
        [μ1],                   # μ: hist markup rate
        [],                     # employees: employees in company
        0,                      # L: labor units in company
        0,                      # ΔLᵈ: desired change in labor force
        [w̄],                    # w̄: wage level
        wᴼ,                     # wᴼ: offered wage
        0.0,                    # wᴼ_max: expected offered wage
        0,                      # O: total amount of machines ordered
        [],                     # production queue
        [Q],                    # Q: production quantity
        [],                     # RD: hist R&D expenditure
        [],                     # IM: hist immitation expenditure
        [],                     # IN: hist innovation expenditure
        [D],                    # D: hist demand
        [],                     # HC: hist clients
        [0.0],                  # Π: hist profits
        fill(0.0, 4),           # debt installments
        [f],                    # f: market share
        [],                     # brochure
        [],                     # orders
        balance,                # balance
        curracc                 # current account   
    )
    return kp
end


"""
Checks if innovation is performed, then calls appropriate functions
"""
function innovate_kp!(
    kp::CapitalGoodProducer, 
    global_param, 
    all_kp::Vector{Int}, 
    kp_distance_matrix::Array{Float64},
    w̄::Float64,
    model::ABM,
    )
    
    # TODO include labor demand for researchers

    # Determine levels of R&D, and how to divide under IN and IM
    set_RD_kp!(kp, global_param.ξ, global_param.ν)
    tech_choices = [(kp.A[end], kp.B[end])]

    # Determine innovation of machines (Dosi et al (2010); eq. 4)
    θ_IN = 1 - exp(-global_param.ζ * kp.IN[end])
    if rand(Bernoulli(θ_IN))
        A_t_in = update_At_kp(kp.A[end], global_param)
        B_t_in = update_Bt_kp(kp.B[end], global_param)
        if A_t_in > kp.A[end] || B_t_in > kp.B[end]
            push!(tech_choices, (A_t_in, B_t_in))
        end
    end

    # TODO compute real value innovation like rer98

    # Determine immitation of competitors
    θ_IM = 1 - exp(-global_param.ζ * kp.IM[end])
    if rand(Bernoulli(θ_IM))
        A_t_im, B_t_im = imitate_technology_kp(kp, all_kp, kp_distance_matrix, model)
        if A_t_im > kp.A[end] || B_t_im > kp.B[end]
            push!(tech_choices, (A_t_im, B_t_im))
        end
    end

    choose_technology_kp!(kp, w̄, global_param, tech_choices)
end


"""
Lets kp choose technology
"""
function choose_technology_kp!(
    kp::CapitalGoodProducer,
    w̄::Float64,
    global_param::GlobalParam,
    tech_choices
    )

    # TODO: DESCRIBE
    update_μ_kp!(kp)

    # Make choice between possible technologies
    if length(tech_choices) == 1
        # If no new technologies, keep current technologies
        push!(kp.A, kp.A[end])
        push!(kp.B, kp.B[end])
        c_h = kp.w̄[end]/kp.B[end]
        # p_h = (1 + global_param.μ1) * c_h
        p_h = (1 + kp.μ[end]) * c_h
        push!(kp.c, c_h)
        push!(kp.p, p_h)
    else
        # If new technologies, update price data
        c_h_kp = map(tech -> kp.w̄[end]/tech[2], tech_choices)
        c_h_cp = map(tech -> w̄/tech[1], tech_choices)
        # p_h = map(c -> (1 + global_param.μ1)*c, c_h_kp)
        p_h = map(c -> (1 + kp.μ[end])*c, c_h_kp)
        r_h = p_h + global_param.b * c_h_cp 
        index = argmin(r_h)
        push!(kp.A, tech_choices[index][1])
        push!(kp.B, tech_choices[index][2])
        push!(kp.c, c_h_kp[index])
        push!(kp.p, p_h[index])
    end
end


"""
Creates brochures and sends to potential clients.
"""
function send_brochures_kp!(
    kp::CapitalGoodProducer,
    all_cp::Vector{Int}, 
    global_param,
    model::ABM;
    n_hist_clients=50::Int
    )

    # Set up brochure
    brochure = (kp.id, kp.p[end], kp.c[end], kp.A[end])
    kp.brochure = brochure

    # Send brochure to historical clients
    for cp_id in kp.HC
        cp = model[cp_id]
        push!(cp.brochures, brochure)
    end

    # Select new clients, send brochure
    NC_potential = setdiff(all_cp, kp.HC)

    if length(kp.HC) == 0
        n_choices = n_hist_clients
    else
        n_choices = Int(round(global_param.γ * length(kp.HC)))
    end
    
    # Send brochures to new clients
    NC = sample(NC_potential, n_choices; replace=false)
    for cp_id in NC
        cp = model[cp_id]
        push!(cp.brochures, brochure)
    end 

end


"""
Uses inverse distances as weights for choice competitor to immitate
"""
function imitate_technology_kp(
    kp::CapitalGoodProducer, 
    all_kp::Vector{Int}, 
    kp_distance_matrix, 
    model::ABM
    )::Tuple{Float64, Float64}

    weights = map(x -> 1/x, kp_distance_matrix[kp.kp_i,:])
    idx = sample(all_kp, Weights(weights))
    # TODO: uncomment

    # idx = sample(all_kp)

    A_t_im = model[idx].A[end]
    B_t_im = model[idx].B[end]
    
    return A_t_im, B_t_im
end


"""
Determines new labor productivity of machine produced for cp
"""
function update_At_kp(
    A_last::Float64, 
    global_param::GlobalParam
    )::Float64
    
    κ_A = rand(Beta(global_param.α1, global_param.β1))
    κ_A = global_param.κ_lower + κ_A * (global_param.κ_upper - global_param.κ_lower)

    A_t_in = A_last * (1 + κ_A)
    return A_t_in
end


"""
Determines new labor productivity of own production method.
"""
function update_Bt_kp(
    B_last::Float64, 
    global_param::GlobalParam
    )::Float64

    κ_B = rand(Beta(global_param.α1, global_param.β1))
    κ_B = global_param.κ_lower + κ_B * (global_param.κ_upper - global_param.κ_lower)

    B_t_in = B_last*(1 + κ_B)
    return B_t_in
end


function set_price_kp!(
    kp::CapitalGoodProducer, 
    μ1::Float64, 
    )

    c_t = kp.w̄[end] / kp.B[end]
    # p_t = (1 + μ1) * c_t
    p_t = (1 + kp.μ[end]) * c_t
    push!(kp.c, c_t)
    push!(kp.p, p_t)
end


"""
Determines the level of R&D, and how it is divided under innovation (IN) 
and immitation (IM). based on Dosi et al (2010)
"""
function set_RD_kp!(
    kp::CapitalGoodProducer, 
    ξ::Float64, 
    ν::Float64
    )

    # Determine total R&D spending at time t, (Dosi et al, 2010; eq. 3)
    prev_S = kp.p[end] * kp.Q[end]
    RD_new = ν * prev_S
    # RD_new = max(ν * kp.Π[end], 0)

    # TODO: now based on prev profit to avoid large losses. If kept, describe!

    push!(kp.RD, RD_new)
    kp.curracc.TCI += RD_new

    # Decide fractions innovation (IN) and immitation (IM), 
    # (Dosi et al, 2010; eq. 3.5)
    IN_t_new = ξ * RD_new
    IM_t_new = (1 - ξ) * RD_new
    push!(kp.IN, IN_t_new)
    push!(kp.IM, IM_t_new)
end


"""
Lets kp receive orders, adds client as historical clients if it is not yet.
"""
function receive_order_kp!(
    kp::CapitalGoodProducer,
    cp_id::Int,
    order
    )

    push!(kp.orders, order)

    # If cp not in HC yet, add as a historical client
    if cp_id ∉ kp.HC
        push!(kp.HC, cp_id)
    end
end


"""
Based on received orders, sets labor demand to fulfill production.
"""
function plan_production_kp!(
    kp::CapitalGoodProducer,
    model::ABM
    )

    update_w̄_p!(kp, model)
    
    # Determine total amount of machines to produce
    if length(kp.orders) > 0
        kp.O = sum(map(order -> order[2] / kp.p[end], kp.orders)) 
    else
        kp.O = 0.0
    end

    # Determine amount of labor to hire
    kp.ΔLᵈ = 50 * floor(max(kp.O / kp.B[end] + kp.RD[end] / kp.w̄[end] - kp.L, -kp.L) / 50)
    # kp.ΔLᵈ = max(kp.O / kp.B[end] + kp.RD[end] / kp.w̄[end] - kp.L, -kp.L)

    # println("labor ", kp.ΔLᵈ, " ", kp.O / kp.B[end], " ", kp.RD[end] / kp.w̄[end], " ", kp.L)
    # kp.ΔLᵈ = kp.O / kp.B[end] - kp.L

    update_wᴼ_max_kp!(kp)
end


"""
Lets kp produce ordered goods based on actual available labor stock.
"""
function produce_goods_kp!(
    kp::CapitalGoodProducer
    )

    total_Q = 0

    # Check if enough labor available for all machines
    unocc_L = kp.L
    for order in kp.orders

        # Only push order in production queue if enough labor available
        q = order[2] / kp.p[end]
        req_L = q / kp.B[end]

        if unocc_L > req_L
            # Full order can be fulfilled
            total_Q += q
            push!(kp.prod_queue, order)
            unocc_L -= req_L
        elseif unocc_L > 0.0
            # Partial order can be fulfilled
            part_q = unocc_L * kp.B[end]
            total_Q += part_q
            new_Iₜ = part_q * kp.p[end]
            push!(kp.prod_queue, (order[1], new_Iₜ))
            unocc_L = 0.0
        end
    end

    # Reset order amount back to zero
    # kp.O = 0

    # println(length(kp.orders), " ", length(kp.prod_queue), " ", kp.L, " ", kp.O)

    # Push production amount
    push!(kp.Q, total_Q)
end


"""
Sends orders from production queue to cp.
"""
function send_orders_kp!(
    kp::CapitalGoodProducer,
    model::ABM
    )

    tot_freq = 0
    for order in kp.prod_queue

        cp_id = order[1]
        Iₜ = order[2]
        freq = Iₜ / kp.p[end]

        machine = initialize_machine(freq; η=0, p=kp.p[end], A=kp.A[end])

        tot_freq += freq

        receive_machines!(model[cp_id], machine, Iₜ)
    end
    
    # Add demand and sales.
    D = tot_freq
    push!(kp.D, D)

    kp.curracc.S = tot_freq * kp.p[end]

    # TODO: request money for machines that were produced

    # TODO: describe how labor market frictions affect delivery machines

    # Empty order and production queue
    reset_order_queue_kp!(kp)
    reset_prod_queue_kp!(kp)
end


function reset_order_queue_kp!(
    kp::CapitalGoodProducer
    )

    kp.orders = []
end


function reset_prod_queue_kp!(
    kp::CapitalGoodProducer
    )

    kp.prod_queue = []
end


function select_HC_kp!(
    kp::CapitalGoodProducer, 
    all_cp::Vector{Int};
    n_hist_clients=10::Int
    )

    kp.HC = sample(all_cp, n_hist_clients; replace=false)
end


function update_wᴼ_max_kp!(
    kp::CapitalGoodProducer
    )
    # TODO: DESCRIBE IN MODEL
    kp.wᴼ_max = kp.B[end] * kp.p[end] 
    # if kp.ΔLᵈ > 0
    #     kp.wᴼ_max = (kp.O * kp.p[end] - kp.w̄[end] * kp.L) / kp.ΔLᵈ
    # else
    #     kp.wᴼ_max = 0
    # end
end


"""
Filters out historical clients if they went bankrupt
"""
function remove_bankrupt_HC_kp!(
    kp::CapitalGoodProducer,
    bankrupt_lp::Vector{Int},
    bankrupt_bp::Vector{Int}
    )

    filter!(bp_id -> bp_id ∉ bankrupt_bp, kp.HC)
    filter!(lp_id -> lp_id ∉ bankrupt_lp, kp.HC)
end


"""
Updates market share of all kp.
"""
function update_marketshare_kp!(
    all_kp::Vector{Int},
    model::ABM
    )

    kp_market = sum(kp_id -> model[kp_id].D[end], all_kp)

    for kp_id in all_kp
        if kp_market == 0
            f = 1 / length(all_kp)
        else
            f = model[kp_id].D[end] / kp_market
        end
        push!(model[kp_id].f, f)
    end
end

# TRIAL: DESCRIBE
function update_μ_kp!(
    kp::CapitalGoodProducer
    )

    b = 0.3
    l = 4

    if length(kp.μ) > l && length(kp.Π) > l && kp.Π[end] != 0
        mean_μ = mean(kp.μ[end-l:end-1])
        # Δμ = (cp.μ[end] - cp.μ[end-1]) / cp.μ[end-1]
        Δμ = (kp.μ[end] - mean_μ) / mean_μ

        mean_Π = mean(kp.Π[end-l:end-1])
        # ΔΠ = (cp.Π[end] - cp.Π[end-1]) / cp.Π[end-1]
        ΔΠ = (kp.Π[end] - mean_Π) / mean_Π

        # println("$mean_μ, $mean_Π")
        # println("Δμ: $Δμ, $(sign(Δμ)), ΔΠ: $ΔΠ, $(sign(ΔΠ))")

        shock = rand(Uniform(0.0, b))

        new_μ = max(kp.μ[end] * (1 + sign(Δμ) * sign(ΔΠ) * shock), 0)
        push!(kp.μ, new_μ)

        # if ΔΠ > 0
        #     new_μ = cp.μ[end] * (1 + Δμ)
        #     push!(cp.μ, new_μ)
        # else
        #     new_μ = max(cp.μ[end] * (1 - Δμ), 0)
        #     push!(cp.μ, new_μ)
        # end

    elseif kp.Π[end] == 0
        push!(kp.μ, kp.μ[end] * (1 + rand(Uniform(-b, 0.0))))
    else
        push!(kp.μ, kp.μ[end] * (1 + rand(Uniform(-b, b))))
    end

end


"""
Replaces bankrupt kp with new kp. Gives them a level of technology and expectations
    from another kp. 
"""
function replace_bankrupt_kp!(
    bankrupt_kp::Vector{Int},
    bankrupt_kp_i::Vector{Int},
    all_kp::Vector{Int},
    φ3::Float64,
    φ4::Float64,
    α2::Float64,
    β2::Float64,
    model::ABM
    )

    # TODO: describe in model

    # Check if not all kp have gone bankrupt, in this case, 
    # kp with highest NW will not be removed
    if length(bankrupt_kp) == length(all_kp)
        kp_id_max_NW = all_kp[1]
        for i in 2:length(all_kp)
            if model[all_kp[i]].curracc.NW > model[kp_id_max_NW].curracc.NW
                kp_id_max_NW = all_kp[i]
            end
        end
        bankrupt_kp = bankrupt_kp[!kp_id_max_NW]
    end

    # Determine all possible kp and set weights for sampling proportional to the 
    # quality of their technology
    poss_kp = filter(kp_id -> kp_id ∉ bankrupt_kp, all_kp)
    weights = map(kp_id -> min(model[kp_id].A[end], model[kp_id].B[end]), poss_kp)

    # Get the technology frontier
    A_max = 0
    B_max = 0
    for kp_id in poss_kp
        if model[kp_id].A[end] > A_max
            A_max = model[kp_id].A[end]
        end

        if model[kp_id].B[end] > B_max
            B_max = model[kp_id].B[end]
        end 
    end

    # Compute the average stock of liquid assets of non-bankrupt kp
    avg_NW = mean(map(kp_id -> model[kp_id].balance.NW, poss_kp))

    # Re-use id of bankrupted company
    for (kp_id, kp_i) in zip(bankrupt_kp, bankrupt_kp_i)

        coeff_NW = rand(Uniform(φ3, φ4))

        # Sample a producer of which to take over the technologies, proportional to the 
        # quality of the technology
        kp_to_copy = model[sample(poss_kp, Weights(weights))]

        # TODO: scale
        a1 = -0.1
        a2 = 0.02
        tech_coeff = a1 + rand(Beta(α2, β2)) * (a2 - a1)
        new_A = A_max * (1 + tech_coeff)
        new_B = B_max * (1 + tech_coeff)

        # Initialize new kp
        new_kp = initialize_kp(
            kp_id, 
            kp_i, 
            length(all_kp);
            NW=coeff_NW * avg_NW,
            A=new_A,
            B=new_B,
            p=kp_to_copy.p[end],
            w̄=kp_to_copy.w̄[end],
            wᴼ=kp_to_copy.wᴼ[end],
            Q=kp_to_copy.Q[end],
            D=kp_to_copy.D[end],
            f=0.0,
            first_gen=false
        )
        add_agent!(new_kp, model)
    end
end