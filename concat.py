"""
Concatenate measures with different parameters in one dataframe
"""


import sys
import pandas as pd

delay_mode_intra = "null"
delay_mode_inter = "unimodal"
intra_params = [1.5]
inter_params = [0.5, 5.0, 10.0]
v_stim = 4.0
metric = "measures"
networks = ["topo", "random"]

# use the homogeneous delay condition as the control condition
dfs = []
# df = pd.read_csv("data/sum/pre/{}_intra=null_inter=null.csv".format(metric))
for network in networks:
    # df = pd.read_csv('data/sum/pre/{}_intra={}_inter={}.csv'.format(metric,delay_mode_intra,delay_mode_inter))
    df = pd.read_csv('data/sum/diff_v_stim/{}_{}_intra=null1.5_inter=null1.5_v_stim={}.csv'.
                     format(metric, network, v_stim), keep_default_na=False)
    df["intra type"] = delay_mode_intra
    df["inter type"] = delay_mode_inter
    df["intra params"] = "null"
    df["inter params"] = "null"
    dfs.append(df)

# add all the other datas
for network_mode in networks:
    for intra_p in intra_params:
        for inter_p in inter_params:
            df = pd.read_csv('data/sum/{}_{}_intra={}{}_inter={}{}.csv'.
                             format(metric, network_mode, delay_mode_intra, intra_p, delay_mode_inter, inter_p),
                             keep_default_na=False)
            df["intra type"] = delay_mode_intra
            df["inter type"] = delay_mode_inter
            df["intra params"] = intra_p
            df["inter params"] = inter_p
            dfs.append(df)

# noise condition does not depend on v_stim
df = pd.read_csv('data/sum/pre/{}_intra={}_inter={}.csv'.format(metric, delay_mode_intra, delay_mode_inter),
                 keep_default_na=False)
df = df[df['network type']=='noise']
dfs.append(df)

# concatenate all dataframes into one and save them
ultimate = pd.concat(dfs, sort=False)
ultimate.to_csv("data/sum/{}_intra={}_inter={}.csv".format(metric, delay_mode_intra, delay_mode_inter), index=False)