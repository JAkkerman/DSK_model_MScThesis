"""
    schedule_per_type(type, shuffle_agents::Bool)
A scheduler that returns multiple schedulers, each for a specific subset 
`shuffle_agents = true` randomizes the order of agents within each group.
"""
function schedule_per_type(model::ABM; shuffle_agents=true::Bool)

    if shuffle_agents
        shuffle!(model.all_hh)
        shuffle!(model.all_cp)
        shuffle!(model.all_kp)
        shuffle!(model.all_p)
    end
    return model.all_hh, model.all_cp, model.all_kp, model.all_p
end