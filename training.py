"""
Use a ridge classifier with cross-validation to fit the state matrix (membrane potential) to correctly classify stimuli.
Compute the classification accuracy and the MSE.
"""
import sys
import time

import numpy as np

from helpers import train, reformat_df
from params import *



"""
parse the parameters and init. data structures
"""
runnum = 2  # default num. of repetition is 5
# TODO: runnum 2 for repetition

network_mode = sys.argv[1]  # input should be either "noise", "random" or "topo"
delay_mode_intra = sys.argv[2]
delay_mode_inter = sys.argv[3]
delay_intra_param = eval(sys.argv[4])
delay_inter_param = eval(sys.argv[5])

# skipping connections
if len(sys.argv) > 6:
    skip_double = bool(eval(sys.argv[6]))  # True if double connection is activated
    delay_skip_param = eval(sys.argv[7])  # increasing delays
    skip_weights = eval(sys.argv[8])  # might want to use decreasing weights, as a factor


accuracy_train = np.zeros((runnum, module_depth))  # accuracy for each trial and each module within a trial
MSE_train = np.zeros((runnum, module_depth))  # averaged mean squared error



"""
load all raw data in appropriate formats
"""
for runindex in range(runnum):
    if len(sys.argv) > 6:
        volt_values = np.load(PATH + "voltvalues_run={}_{}_intra={}{}_inter={}{}_skip_double={}_d={}_w={}.npy".
                              format(runindex, network_mode,
                                     delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param,
                                     skip_double, delay_skip_param, skip_weights))
        stimuli = np.load(PATH + "stimuli_run={}_{}_intra={}{}_inter={}{}_skip_double={}_d={}_w={}.npy".
                          format(runindex, network_mode,
                                 delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param,
                                 skip_double, delay_skip_param, skip_weights))
    else:
        volt_values = np.load(PATH + "voltvalues_run={}_{}_intra={}{}_inter={}{}.npy".
                              format(runindex, network_mode,
                                     delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param))
        stimuli = np.load(PATH + "stimuli_run={}_{}_intra={}{}_inter={}{}.npy".
                          format(runindex, network_mode,
                                 delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param))



    """
    train the classifier and test it
    """
    accuracy_train[runindex], MSE_train[runindex] = train(volt_values=volt_values, target_output=stimuli)
    print("training is done: ", time.process_time())



"""
save the summary data to the file
"""
# reformat the numpy array into data frame
df_acc = reformat_df(network_mode, accuracy_train)
df_mse = reformat_df(network_mode, MSE_train)

# save the data frame
df_acc['MSE'] = df_mse['value']
df_acc.rename(columns={"value": "accuracy"}, inplace=True)

if len(sys.argv) > 6:
    df_acc.to_csv('training_{}_intra={}{}_inter={}{}_skip_double={}_d={}_w={}.csv'.
                  format(network_mode, delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param,
                         skip_double, delay_skip_param, skip_weights), index=False)
else:
    df_acc.to_csv('training_{}_intra={}{}_inter={}{}.csv'.
                  format(network_mode, delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param),
                  index=False)
