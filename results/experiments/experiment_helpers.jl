function compute_growthrates(
    ts::Vector{Float64}
    )

    # Compute growth rates
    growthrates = 100 .* (ts[1:end-1] .- ts[2:end]) ./ ts[2:end]

    # Make sure no Infs or NaNs in ts before returning
    filter!(!isnan, growthrates)
    filter!(!isinf, growthrates)

    return growthrates
end


"""
Computes moments of runoutput and writes to dataframe or array
"""
function convertrunoutput(
    runoutput::RunOutput,
    sim_nr::Int64;
    return_as_df::Bool=false,
    t_warmup::Int64=300,
    )

    # Prepare data to be written to dataframe
    GDP_1st = mean(runoutput.GDP_growth[t_warmup:end])
    GDP_2nd = var(runoutput.GDP_growth[t_warmup:end])
    GDP_3rd = skewness(runoutput.GDP_growth[t_warmup:end])
    GDP_4th = kurtosis(runoutput.GDP_growth[t_warmup:end])

    acorr_GDP = cor(runoutput.GDP_growth[t_warmup+1:end], runoutput.GDP_growth[t_warmup:end-1])

    # Write unemployment data to dataframe
    U_1st = mean(runoutput.U[t_warmup:end])
    U_2nd = var(runoutput.U[t_warmup:end])

    dU = compute_growthrates(runoutput.U[t_warmup:end])
    dU_1st = mean(dU[t_warmup:end])
    dU_2nd = var(dU[t_warmup:end])
    dU_3rd = skewness(dU[t_warmup:end])
    dU_4th = kurtosis(dU[t_warmup:end])
    # corr_GDP_dU = cor(runoutput.GDP_growth[t_warmup:end], runoutput.dU[t_warmup:end])

    dI = compute_growthrates(runoutput.I)
    dI_1st = mean(dI[t_warmup:end])
    dI_2nd = var(dI[t_warmup:end])

    dC = compute_growthrates(runoutput.C)
    dC_1st = mean(dC[t_warmup:end])
    dC_2nd = var(dC[t_warmup:end])

    # Write Gini data to dataframe
    GINI_I_1st = mean(runoutput.GINI_I[t_warmup:end])
    GINI_I_2nd = var(runoutput.GINI_I[t_warmup:end])

    GINI_W_1st = mean(runoutput.GINI_W[t_warmup:end])
    GINI_W_2nd = var(runoutput.GINI_W[t_warmup:end])

    # Write poverty data to dataframe
    # FGT_1st = mean(runoutput.FGT[t_warmup:end])
    # FGT_2nd = var(runoutput.FGT[t_warmup:end])

    # Write average bankrupcy rate
    bankr_1st = mean(runoutput.bankrupty_cp[t_warmup:end])
    bankr_2nd = var(runoutput.bankrupty_cp[t_warmup:end])

    # Add productivity growth
    LP_g = compute_growthrates(runoutput.avg_??_LP)
    LP_g_1st = mean(LP_g[t_warmup:end])
    LP_g_2nd = var(LP_g[t_warmup:end])

    EE_g = compute_growthrates(runoutput.avg_??_EE)
    EE_g_1st = mean(EE_g[t_warmup:end])
    EE_g_2nd = var(EE_g[t_warmup:end])

    EF_g = compute_growthrates(runoutput.avg_??_EF)
    EF_g_1st = mean(EF_g[t_warmup:end])
    EF_g_2nd = var(EF_g[t_warmup:end])

    # Write emissions indexes
    em2030 = runoutput.emissions_index[t_warmup + 120]
    em2040 = runoutput.emissions_index[t_warmup + 240]
    em2050 = runoutput.emissions_index[t_warmup + 360]

    if return_as_df
        return DataFrame(
                    :sim_nr => sim_nr,
                    :GDP_1st => GDP_1st,
                    :GDP_2nd => GDP_2nd,
                    :GDP_3rd => GDP_3rd,
                    :GDP_4th => GDP_4th,
                    :acorr_GDP => acorr_GDP,
                    :U_1st => U_1st,
                    :U_2nd => U_2nd,
                    :dU_1st => dU_1st,
                    :dU_2nd => dU_2nd,
                    :dU_3rd => dU_3rd,
                    :dU_4th => dU_4th,
                    # :corr_GDP_dU => corr_GDP_dU,
                    :dI_1st => dI_1st,
                    :dI_2nd => dI_2nd,
                    :dC_1st => dC_1st,
                    :dC_2nd => dC_2nd, 
                    :GINI_I_1st => GINI_I_1st,
                    :GINI_I_2nd => GINI_I_2nd,
                    :GINI_W_1st => GINI_W_1st,
                    :GINI_W_2nd => GINI_W_2nd,
                    :bankr_1st => bankr_1st,
                    :bankr_2nd => bankr_2nd,
                    :LP_g_1st => LP_g_1st,
                    :LP_g_2nd => LP_g_2nd,
                    :EE_g_1st => EE_g_1st,
                    :EE_g_2nd => EE_g_2nd,
                    :EF_g_1st => EF_g_1st,
                    :EF_g_2nd => EF_g_2nd,
                    # :FGT_1st => FGT_1st,
                    # :FGT_2nd => FGT_2nd,
                    :em2030 => em2030,
                    :em2040 => em2040,
                    :em2050 => em2050
                )
    else
        return [sim_nr,
                GDP_1st, GDP_2nd, GDP_3rd, GDP_4th, acorr_GDP,
                U_1st, U_2nd, dU_1st, dU_2nd, dU_3rd, dU_4th, 
                # corr_GDP_dU,
                dI_1st, dI_2nd, dC_1st, dC_2nd, 
                GINI_I_1st, GINI_I_2nd, GINI_W_1st, GINI_W_2nd, 
                bankr_1st, bankr_2nd,
                LP_g_1st, LP_g_2nd, EE_g_1st, EE_g_2nd,
                EF_g_1st, EF_g_2nd, 
                # FGT_1st, FGT_2nd,
                em2030, em2040, em2050
                ]
    end
end


function savefulloutput(
    runoutput::RunOutput,
    sim_nr::Int64;
    return_as_df::Bool=false,
    t_warmup::Int64=300,
    )

    data = vcat(
        [sim_nr],
        runoutput.GDP_growth[t_warmup:end],
        runoutput.U[t_warmup:end],
        runoutput.C[t_warmup:end],
        runoutput.I[t_warmup:end],
        # runoutput.GINI_I[t_warmup:end],
        # runoutput.GINI_W[t_warmup:end],
        runoutput.emissions_index[t_warmup:end]
    )

    if return_as_df

        cols_GDP = map(i -> Symbol("GDP_$i"), t_warmup:t_warmup+360)
        cols_U = map(i -> Symbol("U_$i"), t_warmup:t_warmup+360)
        cols_C = map(i -> Symbol("C_$i"), t_warmup:t_warmup+360)
        cols_I = map(i -> Symbol("I_$i"), t_warmup:t_warmup+360)
        # cols_GINI_I = map(i -> Symbol("GINI_I_$i"), t_warmup:460)
        # cols_GINI_W = map(i -> Symbol("GINI_W_$i"), t_warmup:460)
        cols_em = map(i -> Symbol("em_$i"), t_warmup:t_warmup+360)

        # cols = vcat([Symbol("sim_nr")], cols_GDP, cols_U, cols_GINI_I, cols_GINI_W, cols_em)

        cols = vcat([Symbol("sim_nr")], cols_GDP, cols_U, cols_C, cols_I, cols_em)

        data = map(d -> [d], data)
        df = DataFrame(data, cols)

        return df
    end

    return data
end