import sys
import time

import numpy as np

from helpers import train, load_data, reshape_arr, reformat_df
from params import *



"""
parse the parameters and init. data structures
"""
runnum = 5  # default num. of repetition is 5
network_mode = sys.argv[1]  # input should be either "noise", "random" or "topo"
delay_mode_intra = 0
delay_mode_inter = 0
delay_intra_param = 0
delay_inter_param = 0
if len(sys.argv) > 3:
    delay_mode_intra = sys.argv[2]
    delay_mode_inter = sys.argv[3]
    delay_intra_param = eval(sys.argv[4])
    delay_inter_param = eval(sys.argv[5])

accuracy_train = np.zeros((runnum, module_depth))
MSE_train = np.zeros((runnum, module_depth))



"""
load all raw data in appropriate formats
"""
for runindex in range(runnum):
    # membrane potential
    expression = "volt_run={}_{}_intra={}{}_inter={}{}*.dat"\
        .format(runindex, network_mode, delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param)
    myarr = np.array(load_data(PATH, expression))
    print("how's my array: ", myarr.shape)
    np.save("myarr", myarr)
    exit()
    volt_values = reshape_arr(np.hstack(np.array(load_data(PATH, expression))))  # samplenum x mod_i x neuronnum
    print("data is loaded: ", time.process_time())



    """
    train the classifier and test it
    """
    accuracy_train[runindex], MSE_train[runindex] = train(volt_values=volt_values,
                              target_output=np.loadtxt(PATH + "stimuli_{}_run={}".format(network_mode, runindex)))
    print("training is done: ", time.process_time())



"""
save the summary data to the file
"""
# reformat the numpy array into data frame
df_acc = reformat_df(network_mode, accuracy_train)
df_mse = reformat_df(network_mode, MSE_train)

# save the data frame
df_acc.to_pickle("accuracy_train_{}_intra={}{}_inter={}{}.p".
                 format(network_mode, delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param))
df_mse.to_pickle("MSE_train_{}_intra={}{}_inter={}{}.p".
                 format(network_mode, delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param))
