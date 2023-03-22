@with_kw struct InitParam
    
    # Agent counts
    n_kp::Int64 = 40                      # number of kp
    n_cp::Int64 = 200                     # number of cp
    n_hh::Int64 = 2500                    # number of hh

    # Init rates
    init_unempl_rate::Float64 = 0.05      # initial unemployment rate

    # Init params of hh
    α₀::Float64 = 0.8                     # initial absolute propensity to consume
    L₀::Float64 = 100.                    # number of labor units
    w₀::Float64 = 1.                      # intial (satisfying and requested) wage
    W₀::Float64 = 200.                    # initial (real) wealth level
    skill_mean::Float64 = 0.              # mean value for lognormal skill distribution
    skill_var::Float64 = 0.75             # variance value for lognormal skill distribution
    βmin::Float64 = 0.6                  # minimum value discount factor
    βmax::Float64 = 0.9                  # maximum value discount factor

    # Init params of cp
    D₀_cp::Float64 = 2000.                # initial level of demand
    w₀_cp::Float64 = 1.                   # intial wage level
    f₀_cp::Float64 = 1/n_cp               # intial market share
    NWₒ_cp::Float64 = 1500.               # initial level of liquid assets for cp

    # Init params of kp
    A_LP_0::Float64 = 1.0                 # initial productivity level A_LP
    A_EE_0::Float64 = 1.0                 # initial productivity level A_EE
    A_EF_0::Float64 = 1.0                 # initial productivity level A_EF
    B_LP_0::Float64 = 1.0                 # initial productivity level B_LP
    B_EE_0::Float64 = 1.0                 # initial productivity level B_EE
    B_EF_0::Float64 = 1.0                 # initial productivity level B_EF

    # Init params of ep
    n_powerplants_init::Int64 = round(Int64, D₀_cp * 1.2 * n_cp) # number of unit of power plants in ep
    frac_green::Float64 = 0.1             # fraction of initial power plants that are green
    markup_ep::Float64 = 0.01             # markup rate energy producer
    Aᵀ_0::Float64 = 1.0                   # initial thermal efficiency
    emᵀ_0::Float64 = 1.0                  # initial emission level
    IC_g_0::Float64 = 12                  # initial fixed costs of green plant investments
end