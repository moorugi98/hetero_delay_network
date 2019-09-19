import sys
import pandas as pd

delay_mode_intra = sys.argv[1]
delay_mode_inter = sys.argv[2]
delay_intra_params = eval(sys.argv[3])
delay_inter_params = eval(sys.argv[4])

dfs = []
for network_mode in ["topo", "random", "noise"]:
    df = pd.read_pickle("/data/sum/reproduce_Barna/real/measures_{}.p".format(network_mode))
    df["intra type"] = delay_mode_intra
    df["inter type"] = delay_mode_inter
    df["intra params"] = "null"
    df["inter params"] = "null"
    dfs.append(df)
for network_mode in ["topo", "random", "noise"]:
    for std in delay_intra_params:  # TODO: change to delay_params_intra and add inter
        df = pd.read_pickle("measures_{}_intra={}{}_inter={}{}.p".  # save the data
                     format(network_mode, delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param))
        df["intra type"] = delay_mode_intra
        df["inter type"] = delay_mode_inter
        df["intra params"] = std
        df["inter params"] = 1.5
        dfs.append(df)
ultimate = pd.concat(dfs).reset_index()
ultimate.to_pickle("measures_intra={}_inter={}.p".format(delay_mode_intra, delay_mode_inter))