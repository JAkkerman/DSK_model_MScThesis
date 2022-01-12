
mutable struct Household <: AbstractAgent
    id :: Int                   # global id
    hh_id :: Int                # hh id
    employed :: Bool            # is employed
    employer :: AbstractAgent   # employer
    I :: Array{Float64}         # hist income
    Iᵉ :: Float64               # expected income
    L :: Float64                # labor units in household
    S :: Array{Float64}         # savings
    B :: Float64                # budget
    w :: Float64                # wage
    ωI :: Float64               # memory param income expectation

end


function pick_cp(hh)

end


"""
Determines consumption budget B
    - Determine income at time t
    - Estimate future income
    - Determine savings rate
    - Set consumption budget
"""
function set_budget_hh!(hh, UB, U, r)
    
    # determine income
    Iₜ = UB
    if (hh.employed)
        Iₜ = hh.w * hh.L
    end
    push!(hh.I, Iₜ)

    hh.Iᵉ = compute_exp_income(hh, U, r)

    # determine savings rate
    s = (Iₜ - Iᵉ) / Iₜ

end


"""
Determines the optimal consumption package
    - Choose which goods to buy
    - Choose optimal consumption package for both goods
"""
function set_cons_package_hh!(hh)

end


function compute_exp_income(hh, U, r)

    ξ = 0
    if (length(U) > 2)
        if U[end] > U[end-1]
            ξ = rand(0, U[end] - U[end-1])
        else
            ξ = rand(U[end] - U[end-1], 0)
        end
    end

    Iᵉ = ωI * hh.Iᵉ + (1 - ωI) * (2 * hh.I[end] - hh.I[end-1]) + ξ * hh.I[end] + r[end] * hh.S[end]
    return Iᵉ
end
