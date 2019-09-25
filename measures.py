"""
compute different metrics to quantify different properties of the network
"""

import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import pairwise_corr, revised_local_variation, avg_firing_rate, fano_factor, load_data
from params import *



"""
parse the parameters and initalise data structures
"""
runnum = 5  # default num. of trial repetition is 5
network_mode = sys.argv[1]  # input should be either "noise", "random" or "topo"
sim_time = eval(sys.argv[2])

if len(sys.argv) > 3:
    delay_mode_intra = sys.argv[3]
    delay_mode_inter = sys.argv[4]
    delay_intra_param = eval(sys.argv[5])
    delay_inter_param = eval(sys.argv[6])

measures = np.zeros((runnum, module_depth, 4))  # four different metrics



"""
raster plot here because I don't want to load spike data in plot.py
"""
# for some reasons I cannot import this function from another script, so just define it here
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
    a1.set(xlabel="time (ms)")
    bin_size = 5  # in msec
    bins = np.arange(plot_time[0], plot_time[1]+0.00066, bin_size)
    heights, edges = np.histogram(spike_times, bins)
    normed = (heights/N)*(1000/bin_size)
    a1.bar(bins[:-1], normed, width=10.0, color="orange", align="edge")
    plt.savefig(filename, bbox_to_inches="tight")



"""
load all raw data in appropriate formats
"""
for runindex in range(runnum):
    # when spikes are saved directly to the file
    # # spikes
    # spike_times = []
    # spike_senders = []
    # for mod_i in range(module_depth):
        # expression = "spike_run={}_{}_intra={}{}_inter={}{}-4000{}-*.dat".\
        #     format(runindex, network_mode, delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param, mod_i+1)
        # print("expression: ", expression)
        # spike_arr = np.vstack(np.array(load_data(PATH, expression)))
        # spike_times.append(spike_arr[:, 1])
        # spike_senders.append(spike_arr[:, 0])
    #     spike_times.append()
    #     spike_senders.append()
    # print("data is lock & loaded: ", time.process_time())



    """
    compute useful measures to test the network
    """
    for mod_i in range(module_depth):
        # load the spikes data
        times = np.load(PATH + 'spiketimes_run={}_{}_intra={}{}_inter={}{}.npy'.
        format(runindex, network_mode, delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param),
                                   allow_pickle=True)[mod_i]
        gids = np.load(PATH + 'spikesenders_run={}_{}_intra={}{}_inter={}{}.npy'.
        format(runindex, network_mode, delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param),
                                     allow_pickle=True)[mod_i]
        if mod_i == 2:
            plot_raster(filename=os.getcwd() + "/figs/rasterplot/{}_intra={}{}_inter={}{}.pdf".
                        format(network_mode, delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param),
                        spike_times=times, spike_senders=gids, layer=mod_i,
                        num_to_plot=500)

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
out_arr = np.column_stack((np.tile(np.arange(n),m), measures.reshape(m*n, -1)))  # reshaping array to dataframe
df_measure = pd.DataFrame(out_arr,
                          columns=["module index", "synchrony", "irregularity", "firing rate", "variability"])
df_measure["network type"] = network_mode
df_measure.to_pickle("measures_{}_intra={}{}_inter={}{}.p".  # save the data
                     format(network_mode, delay_mode_intra, delay_intra_param, delay_mode_inter, delay_inter_param))



