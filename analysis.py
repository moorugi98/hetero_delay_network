####
# script to compute useful measures such as correlation, firing rate and to plot the result
####

import itertools
import random
import numpy as np
import matplotlib.pyplot as pylab
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split

from params import *


# I. synchrony
def pairwise_corr(spike_times, spike_senders, record_time):
    """
    compute the avg. Pearson pairwise_corr for randomly selected pair of spike trains
    : param spike_times: list, nest.GetStatus(spike_detector, ["event"])[0]["times"]
    :param spike_senders: list, nest.GetStatus(spike_detector, ["event"])[0]["senders"]
    : param record_time: int, recording time in ms
    :return: int, avg. Pearson pairwise_corr
    """

    # parameter
    sum_pearson_coef = 0
    num_pair = 500  # num. of pair to compute avg. pairwise_cor
    bin_size = 2  # bin_size for computing spike train

    spike_times = spike_times - t_onset  # control for t_onset ruining time bin

    pairs = list(itertools.combinations(np.unique(spike_senders), 2))
    for pair in random.sample(pairs, num_pair): # iterate over random num_pair of pairs
        boolean_arr = np.zeros((2, int(record_time // bin_size)), dtype=bool) # init spike train
        for nid, neuron in enumerate(pair): # iterate over two neurons in each pair
            indices = np.where(neuron == spike_senders)[0] # [12, 17, 21,...] indices of spike time of a current neuron
            st = spike_times[indices] - 0.00001 # [0.9999, 18.999, 238.9999...] # dirty trick to make binning easier
            boolean_arr[nid, np.int_(st//bin_size)] = True # now the array is full with binned spike train
        sum_pearson_coef += np.corrcoef(boolean_arr)[0,1] # compute sum of Pearson corr. coef.
    return sum_pearson_coef / num_pair


# II. LvR
def revised_local_variation(spike_times, spike_senders):
    """
    compute the revised_local_variation (LvR) suggested by Shinomoto neuron-wise and return the avg. value
    :param spike_times: list, nest.GetStatus(spike_detector, ["event"])[0]["times"]
    :param spike_senders: list, nest.GetStatus(spike_detector, ["event"])[0]["senders"]
    :return: int, mean LvR value
    """
    neuron_list = np.unique(spike_senders)  # all unique gids of neurons
    print("LvR;neuron_list: ", neuron_list[:10])
    lvr = np.zeros(neuron_list.shape[0])  # save lvr for each neuron

    for ni, neuron in enumerate(neuron_list):
        isi = np.ediff1d(spike_times[neuron == spike_senders])  # inter spike interval
        if isi.shape[0] < 2:
            lvr[ni] = np.nan
        else:
            lvr[ni] = ((3 / isi[:-1].shape[0]) *
            (1 - np.sum(4*isi[:-1]*isi[1:]) / np.sum((isi[:-1] + isi[1:])**2)) *
            (1 + (4*tau_ref) / np.sum(isi[:-1] + isi[1:])))
    return np.nanmean(lvr)  # return the avg. value over all neurons


# III. firing rate
def avg_firing_rate(spike_senders, record_time, N):
    """
    compute the avg.firing rate over the whole population and time
    :param spike_senders: list, nest.GetStatus(spike_detector, ["event"])[0]["senders"]
    :param record_time: int, length of recording in ms
    :param N: int, num. of neurons to record from
    :return: int, avg. firing rate
    """

    record_time = record_time / 1000
    return spike_senders.shape[0] / (record_time * N)


# IV. Fano factor
def fano_factor(spike_times, record_time):
    """
    compute the Fano factor by using the population avg. firing rate with 10 ms bin.
    :param spike_times: list, nest.GetStatus(spike_detector, ["event"])[0]["times"]
    :param record_time: int, recording time in ms
    :return: Fano factor
    """
    bin_size = 10  # width of a  single bin in ms
    bins = np.arange(0, record_time+0.1, bin_size)  # define bin edges
    hist, edges = np.histogram(spike_times, bins=bins)
    print("fano; hist: ", hist)
    return np.var(hist) / np.mean(hist)


# V. classification
def train(volt_values, target_output):
    """
    function to train a simple linear regression to fit the snapshot of membrane potential to binary classification
    using a ridge regression with cross-validation for regularization parameter.
    :param volt_values: np.arr, shape: len.of stim. presentation x N_E.
    snapshots of membrane potential at each stimuli offset.
    :param target_output: np.arr, shape: num. of stimuli x len. of stim. presentation. @sym_seq in the main.py
    :return: list, each element saves the score for each module
    """
    scores = []
    for mod_i in range(module_depth):
        X = volt_values[:,mod_i,:]  # take only activities of exci. neurons. 50x8000
        print("X in the training with shape timestep x neuronnum.: ", X.shape)
        # split the data
        split_ratio = 0.2  # how much percentage of the data will be used for the test
        X_train, X_test, y_train, y_test = train_test_split(X, target_output, test_size=split_ratio)

        # fit
        deltas = [0, 0.1, 1.0, 10.0, 100.0]  # regularization parameter
        fit_model = lm.RidgeClassifierCV(alpha=deltas, fit_intercept=True)\
            .fit(X=X_train, y=y_train) # linear ridge regression with cross-validation for regularization parameter

        # test
        scores.append(fit_model.score(X_test, y_test))
    return scores



#####################################################
# Plotting
#####################################################

def plot_V_m(filename, times, voltages, num_to_plot=5):
    """
    plot the membrane potetinal time trace by selecting random neurons and plot the avg. at the end.
    :filename: str, name of the file to save
    :param times: list, nest.GetStatus(voltage_detector, ["event"])[0]["times"] and reshaped to avoid repetitiveness
    :param voltages: list, nest.GetStatus(voltage_detector, ["event"])[0]["V_m"]
    :param num_to_plot: int, select how much neurons should be plotted from the population
    :return: None
    """
    selected = np.random.choice(np.arange(0,N), num_to_plot) # for each module select num_to_plot neurons
    fig, axes = pylab.subplots(nrows=num_to_plot+1, ncols=1)
    for neuron_index in range(num_to_plot):
        axes[neuron_index].plot(times, voltages[:,neuron_index], "gray")
    axes[-1].plot(times, np.mean(voltages, axis=1), "blue") # plot the avg.
    pylab.xlabel("time(ms)", fontsize=30)
    pylab.ylabel("potential(mV)", fontsize=30)
    pylab.savefig(filename, bbox_to_inches="tight")


def plot_raster(filename, spike_times, spike_senders, layer, num_to_plot = 1000, plot_time = [6000, 8000]):
    """
    make a raster plot with @num_to_plot neurons
    :filename: str, name of the file to save
    :param spike_times: list, nest.GetStatus(spike_detector, ["event"])[0]["times"]
    :param spike_senders: list, nest.GetStatus(spike_detector, ["event"])[0]["senders"]
    :param num_to_plot: int, num of neurons to plot, default = 1000
    :param plot_time: list, interval of time to plot the spikes
    :return: None
    """
    spike_times = spike_times[spike_times < plot_time[1]]  #
    spike_times = spike_times[spike_times > plot_time[0]]
    rand_choice = np.random.randint(0 +  N*layer, N*(layer+1), num_to_plot) # choose neurons to plot randomly
    mask = np.isin(spike_senders, rand_choice)
    pylab.scatter(spike_times[mask], spike_senders[mask], s=0.1, c="r")
    pylab.savefig(filename, bbox_to_inches="tight")


def plot_result(filename, arr_to_plot, title, ylabel):
    """
    helper function to plot the result of various measures
    :filename: str, name of the file to save
    :param arr_to_plot: np.arr, the data to plot
    :param title: str, the title of the plot
    :param ylabel: str, ylabel of the plot
    """
    pylab.figure()
    pylab.plot(arr_to_plot)
    pylab.xticks(np.arange(arr_to_plot.shape[0]), ["M0","M1","M2","M3"])
    pylab.ylabel(ylabel)
    pylab.title(title)
    pylab.savefig(filename, bbox_to_inches="tight")
