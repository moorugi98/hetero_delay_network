"""
Concatenate measures with different parameters in one dataframe
"""

import sys
import pandas as pd

delay_mode_intra = "null"
delay_mode_inter = "null"
intra_params = [1.5]
inter_params = [1.5]
v_stim = 4.0
metric = "training"
networks = ["topo", "random"]

# use the homogeneous delay condition as the control condition
dfs = []
# df = pd.read_csv("data/sum/pre/{}_intra=null_inter=null.csv".format(metric))
for network in networks:
    # df = pd.read_csv('data/sum/pre/{}_intra={}_inter={}.csv'.format(metric,delay_mode_intra,delay_mode_inter))
    df = pd.read_csv('data/sum/diff_v_stim/{}_{}_intra=1.5_inter=1.5_v_stim={}.csv'.
                     format(metric, network, v_stim), keep_default_na=False)
    df["intra type"] = delay_mode_intra
    df["inter type"] = delay_mode_inter
    df["intra params"] = "null"
    df["inter params"] = "null"
    df['double conn'] = 'null'
    df['skip delays'] = 'null'
    df['skip weights'] = 'null'
    dfs.append(df)

# add all the other datas
for network_mode in networks:
    for intra_p in intra_params:
        for inter_p in inter_params:
            # df = pd.read_csv('data/sum/{}_{}_intra={}{}_inter={}{}.csv'.
            #                  format(metric, network_mode, delay_mode_intra, intra_p, delay_mode_inter, inter_p),
            #                  keep_default_na=False)
            for skip_double in [str(True), str(False)]:
                for skip_p in [1.5, 3.0]:
                    for skip_w in [1.0, 0.5]:
                        df = pd.read_csv('data/sum/{}_{}_intra={}{}_inter={}{}_skip_double={}_d={}_w={}.csv'.
                                         format(metric, network_mode, delay_mode_intra, intra_p, delay_mode_inter, inter_p, skip_double, skip_p, skip_w))
                        df["intra type"] = delay_mode_intra
                        df["inter type"] = delay_mode_inter
                        df["intra params"] = intra_p
                        df["inter params"] = inter_p
                        df['double conn'] = skip_double
                        df['skip delays'] = skip_p
                        df['skip weights'] = skip_w
                        dfs.append(df)

# # noise condition does not depend on v_stim
# df = pd.read_csv('data/sum/pre/{}_intra={}_inter={}.csv'.format(metric, delay_mode_intra, delay_mode_inter),
#                  keep_default_na=False)
# df = df[df['network type']=='noise']
# dfs.append(df)

# concatenate all dataframes into one and save them
ultimate = pd.concat(dfs, sort=False)
ultimate.to_csv("data/sum/{}_intra={}_inter={}.csv".format(metric, delay_mode_intra, delay_mode_inter), index=False)