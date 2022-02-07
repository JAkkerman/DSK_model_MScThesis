"""
File used to write simulation results to data files
"""

using DataFrames
using CSV

"""
Saves macro variables of interest to csv

Receives:
    macro_struct: mut struct with macro variables of interest
"""
function save_macro_data(macro_struct)

    df = DataFrame(GDP=macro_struct.GDP,
                   GDP_I=macro_struct.GDP_I,
                   GDP_cp=macro_struct.GDP_Π_cp,
                   GDP_kp=macro_struct.GDP_Π_kp,
                   UR=macro_struct.U,
                   Exp_UB=macro_struct.Exp_UB,
                   s_avg=macro_struct.s̄_avg,
                   s_std=macro_struct.s̄_std,
                   w_avg=macro_struct.w̄_avg,
                   w_std=macro_struct.w̄_std,
                   I_avg=macro_struct.Ī_avg,
                   I_std=macro_struct.Ī_std,
                   dL_avg=macro_struct.ΔL̄_avg,
                   dL_std=macro_struct.ΔL̄_std,
                   dL_cp_avg=macro_struct.ΔL̄_cp_avg,
                   dL_kp_avg=macro_struct.ΔL̄_kp_avg)
    CSV.write("results/result_data/first.csv", df)
end