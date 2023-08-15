"""
    shift_and_append!(ts::Vector{Union{Float64, Int64}}, neval::Union{Float64, Int64})

Shifts all elements in the passed array to the left and adds the new
    value to the end.
"""
function shift_and_append!(
    ts::Union{Vector{Float64}, Vector{Int64}},
    newval::Union{Float64, Int64}
    )

    # Shift values
    ts[1:end-1] = ts[2:end]

    # Add new value
    ts[end] = newval
end


"""
Converts global agent id to local id only for agent type. 
    Order of initialization: hh, cp, kp
"""
function convert_global_id(global_id::Int, model::ABM, is_hh::Bool=false, is_cp::Bool=false)
    if is_hh
        return global_id
    elseif is_cp
        return global_id - model.i_param.n_hh
    end
    return global_id - model.i_param.n_hh - model.i_param.n_cp
end