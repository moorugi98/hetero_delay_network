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
    num_pair = 500 # num. of pair to compute avg. pairwise_cor
    bin_size = 2 # bin_size for computing spike train
    
    pairs = list(itertools.combinations(np.unique(spike_senders), 2))
    for pair in random.sample(pairs, num_pair): # iterate over random num_pair of pairs
        boolean_arr = np.zeros((2, int(record_time // bin_size)), dtype=bool) # init spike train
        for nid, neuron in enumerate(pair): # iterate over two neurons in each pair
            indices = np.where(neuron == spike_senders)[0] # [12, 17, 21,...] indices of spike time of a current neuron
            st = spike_times[indices] - 0.0001 # [0.87, 18.2, 238.09...] # dirty trick to make binning easier
            boolean_arr[nid, np.int_(st//bin_size)] = True # now the array is full with binned spike train
        sum_pearson_coef += np.corrcoef(boolean_arr)[0,1] # compute sum of Pearson corr. coef.
    return sum_pearson_coef / num_pair


# II. firing rate
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


# III. Fano factor
def fano_factor(spike_times):
    """
    compute the Fano factor by using the population avg. firing rate with 10 ms bin.
    :param spike_times: list, nest.GetStatus(spike_detector, ["event"])[0]["times"]
    :return: Fano factor
    """
    bin_size = 10
    hist, edges = np.histogram(spike_times, bins=bin_size)
    return np.var(hist) / np.mean(hist)


# IV. classification
def train(voltage_events):
    # arrays
    voltages_array = np.reshape(voltage_events["V_m"], (-1, N_E)) # record only from exci.
    X = voltages_array[t_asterisk::pos_dur] # snapshot of mem.pot. at the end of stimuli input with shape (3,800)
    target_output = 1

    # split the data
    split_ratio = 0.2  # how much percentage of the data will be used for the test
    X_train, X_test, y_train, y_test = train_test_split(X, target_output, test_size=split_ratio)

    # I.
    # fit
    delta = 0.1 # regularization parameter
    fit_model = lm.Ridge(alpha=delta).fit(X=X_train, y=y_train)

    # test
    fit_model.predict(X_test, y_test)

    # II.
    # fit
    deltas = [1,5,10]
    fit_model = lm.RidgeClassifierCV(alpha=deltas, fit_intercept=True)\
        .fit(X=X_train, y=y_train) # linear ridge regression

    # test
    fit_model.score(X_test, y_test)


def plot_V_m(filename, times, voltages, num_to_plot=5):
    """
    plot the membrane potetinal time trace by selecting random neurons and plot the avg. at the end.
    :filename: str, name of the file to save
    :param times: list, nest.GetStatus(voltage_detector, ["event"])[0]["times"] and reshaped to avoid repetitiveness
    :param voltages: list, nest.GetStatus(voltage_detector, ["event"])[0]["V_m"]
    :param num_to_plot: int, select how much neurons should be plotted from the population
    :return: nothin'
    """
    selected = np.random.choice(np.arange(0,N), num_to_plot) # for each module select num_to_plot neurons
    fig, axes = pylab.subplots(nrows=num_to_plot+1, ncols=1)
    for neuron_index in range(num_to_plot):
        axes[neuron_index].plot(times, voltages[:,neuron_index], "gray")
    axes[-1].plot(times, np.mean(voltages, axis=1), "blue") # plot the avg.
    pylab.xlabel("time(ms)", fontsize=30)
    pylab.ylabel("potential(mV)", fontsize=30)
    pylab.savefig(filename, bbox_to_inches="tight")


def plot_raster(filename, spike_times, spike_senders, layer, num_to_plot = 1000):
    """
    make a raster plot with @num_to_plot neurons
    :filename: str, name of the file to save
    :param spike_times: list, nest.GetStatus(spike_detector, ["event"])[0]["times"]
    :param spike_senders: list, nest.GetStatus(spike_detector, ["event"])[0]["senders"]
    :param num_to_plot: int, num of neurons to plot, default = 1000
    :return:
    """
    rand_choice = np.random.randint(0 +  N*layer, N*(layer+1), num_to_plot) # choose neurons to plot randomly
    print("spike_senders: ", spike_senders)
    print("rand_choice: ", rand_choice)
    mask = np.isin(spike_senders, rand_choice)
    print("mask: ", mask)
    print("spike senders: ", spike_senders[mask])
    print("length: ", spike_senders[mask].shape)
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


