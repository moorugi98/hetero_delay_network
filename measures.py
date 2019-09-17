import sys
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import pairwise_corr, revised_local_variation, avg_firing_rate, fano_factor, load_data
from params import *


PATH = os.getcwd() + "/data/raw/hetero/unimodal/intra/"
# PATH = os.getcwd() + "/data/dummy/"
print("path: ", PATH)



"""
parse the parameters and init. data structures
"""
runnum = 5  # default num. of repetition is 5
network_mode = sys.argv[1]  # input should be either "noise", "random" or "topo"
sim_time = eval(sys.argv[2])

delay_mode_intra = 0
delay_mode_inter = 0
delay_intra_param = 0
delay_inter_param = 0
if len(sys.argv) > 3:
    delay_mode_intra = sys.argv[3]
    delay_mode_inter = sys.argv[4]
    delay_intra_param = eval(sys.argv[5])
    delay_inter_param = eval(sys.argv[6])

measures = np.zeros((runnum, module_depth, 4))



"""
load all raw data in appropriate formats
"""



for runindex in range(runnum):
    # spikes
    spike_times = []
    spike_senders = []
    for mod_i in range(module_depth):
        # TODO
        # change the naming convention to be general
        expression = "spike_{}_run={}_std={}-4000{}-*.dat".format(network_mode, delay_intra_param, runindex, mod_i + 1)
        spike_arr = np.vstack(np.array(load_data(PATH, expression)))
        spike_times.append(spike_arr[:, 1])
        spike_senders.append(spike_arr[:, 0])

    print("data is lock & loaded: ", time.process_time())



    """
    compute useful measures to test the network
    """
    for mod_i in range(module_depth):
        times = spike_times[mod_i]
        gids = spike_senders[mod_i]
        measures[runindex, mod_i, :] = [pairwise_corr(spike_times=times, spike_senders=gids, record_time=sim_time),
                                        revised_local_variation(spike_times=times, spike_senders=gids),
                                        avg_firing_rate(spike_senders=gids, record_time=sim_time, N=N),
                                        fano_factor(spike_times=times, record_time=sim_time, N=N)
                                        ]
    print("measures are computed now: ", time.process_time())



"""
save the summary data to the file
"""
m,n,r = measures.shape
out_arr = np.column_stack((np.tile(np.arange(n),m), measures.reshape(m*n, -1)))
df_measure = pd.DataFrame(out_arr,
                          columns=["module index", "synchrony", "irregularity", "firing rate", "variability"])
df_measure["network type"] = network_mode
df_measure.to_pickle("measures_{}_intra={}{}_inter={}{}.p".  # save the data
                     format(network_mode, delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param))



"""
raster plot here because I don't want to load spike data in plot.py
"""
def plot_raster(filename, spike_times, spike_senders, layer, num_to_plot=100, plot_time=(6000, 8000)):
    """
    make a raster plot with @num_to_plot neurons
    :filename: str, name of the file to save
    :param spike_times: list, nest.GetStatus(spike_detector, ["event"])[0]["times"]
    :param spike_senders: list, nest.GetStatus(spike_detector, ["event"])[0]["senders"]
    :param num_to_plot: int, num of neurons to plot, default = 1000
    :param plot_time: list, interval of time to plot the spikes
    :return: None
    """
    fig,(a0, a1)= plt.subplots(nrows=2, ncols=1, figsize=(8,5), gridspec_kw={'height_ratios': [3,1]})
    mask_time = (spike_times <= plot_time[1]) & (spike_times >= plot_time[0])  # choose time to plot
    rand_choice = np.random.randint(0 + N * layer, N * (layer + 1), num_to_plot)
    mask_ids = np.isin(spike_senders, rand_choice)  # choose neurons to plot randomly
    a0.scatter(spike_times[mask_time & mask_ids], spike_senders[mask_time & mask_ids], s=0.1, c="r")
    a0.set(xlabel="time (ms)")
    bin_size = 5  # in msec
    bins = np.arange(plot_time[0], plot_time[1]+0.00066, bin_size)
    heights, edges = np.histogram(spike_times, bins)
    normed = (heights/N)*(1000/bin_size)
    a1.bar(bins[:-1], normed, width=10.0, color="orange", align="edge")
    plt.savefig(filename, bbox_to_inches="tight")

mod_i = 2  # which layer to plot (M2)
plot_raster(filename=os.getcwd() + "/figs/rasterplot/{}_intra={}{}_inter={}{}.pdf".
            format(network_mode, delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param),
            spike_times=spike_times[mod_i], spike_senders=spike_senders[mod_i], layer=mod_i, num_to_plot=500)