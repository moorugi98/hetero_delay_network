"""
Concatenate measures with different parameters in one dataframe
"""


import sys
import pandas as pd

delay_mode_intra = "unimodal"
delay_mode_inter = "null"
intra_params = [0.5, 5.0, 10.0]
inter_params = [1.5]

# use the homogeneous delay condition as the control condition
dfs = []
for network_mode in ["topo", "random", "noise"]:
    df = pd.read_pickle("data/sum/reproduce_Barna/real/measures_{}.p".format(network_mode))
    df["intra type"] = delay_mode_intra
    df["inter type"] = delay_mode_inter
    df["intra params"] = "null"
    df["inter params"] = "null"
    dfs.append(df)

# add all the other datas
for network_mode in ['topo', "random", "noise"]:
    for intra_p in intra_params:
        for inter_p in inter_params:
            df = pd.read_pickle("data/sum/intra={}_inter={}/measures_{}_intra={}{}_inter={}{}.p".
                 format(delay_mode_intra, delay_mode_inter, network_mode,
                        delay_mode_intra, intra_p, delay_mode_inter, inter_p))
            df["intra type"] = delay_mode_intra
            df["inter type"] = delay_mode_inter
            df["intra params"] = intra_p
            df["inter params"] = inter_p
            dfs.append(df)

# concatenate all dataframes into one and save them
ultimate = pd.concat(dfs).reset_index()
ultimate.to_pickle("data/sum/measures_intra={}_inter={}.p".format(delay_mode_intra, delay_mode_inter))