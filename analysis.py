####
# script to compute useful measures such as correlation, firing rate and to plot the result
####

import itertools
import random
import numpy as np
import matplotlib.pyplot as pylab
from scipy.stats import gaussian_kde
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split

from params import *
import time

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

    pairs = random.sample(list(itertools.combinations(np.unique(spike_senders), 2)), num_pair)  # num_pair rand. pairs
    for pair in pairs:
        boolean_arr = np.zeros((2, int(record_time // bin_size)), dtype=bool)  # init spike train
        for nid, neuron in enumerate(pair):  # iterate over two neurons in each pair
            indices = np.where(neuron == spike_senders)[0]  # [12, 17, 21,...] indices of spike time of a current neuron
            st = spike_times[indices] - 0.00001  # [0.9999, 18.999, 238.9999...] # dirty trick to make binning easier
            boolean_arr[nid, np.int_(st//bin_size)] = True  # now the array is full with binned spike train
        sum_pearson_coef += np.corrcoef(boolean_arr)[0,1]  # compute sum of Pearson corr. coef.
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
    hist = hist * 1000/bin_size
    return np.var(hist) / np.mean(hist)


def fano_factor_normed(spike_times, record_time):
    bin_size = 10  # width of a single bin in ms
    bins = np.arange(0, record_time + 0.1, bin_size)  # define bin edges
    hist, edges = np.histogram(spike_times, bins=bins)
    hist = hist * 1000 / bin_size
    # compute firing rate using Gaussian kernel on top of histogram and save value for every tick (ms)
    tick = 0.1
    firingrate = gaussian_kde(dataset=hist).evaluate(np.arange(0, record_time+tick, tick))
    return np.var(firingrate) / np.mean(firingrate)




# V. classification
def train(volt_values, target_output, split_ratio=0.5):
    """
    function to train a simple linear regression to fit the snapshot of membrane potential to binary classification
    using a ridge regression with cross-validation for regularization parameter.
    :param volt_values: np.arr, shape: len.of stim. presentation x N_E.
    snapshots of membrane potential at each stimuli offset.
    :param target_output: np.arr, shape: num. of stimuli x len. of stim. presentation. @sym_seq in the main.py
    :param split_ratio: float, percentage of the data to be used for the test
    :return: list, saves the score for each module
    """
    print("before train: ", time.process_time())
    scores = []  # array to save accuracy score for each module
    MSE = []
    for mod_i in range(module_depth):
        # split the data into training and test sets
        # X_train dim: #sample(timesteps) * (1-split_ratio) x #features(neurons)
        # y_train dim: #sample * #classes(stimuli)
        X_train, X_test, y_train, y_test = train_test_split(volt_values[:, mod_i, :],  # for each module
                                                            np.transpose(np.int_(target_output)), test_size=split_ratio)

        # linear ridge regression with cross-validation for regularization parameter
        deltas = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]  # regularization parameter
        fit_model = lm.RidgeClassifierCV(alphas=deltas, fit_intercept=True, store_cv_values=True)\
            .fit(X=X_train, y=y_train)

        # compute the output using the trained weight and test dataset with winner-take-all prediction (hard decision)
        # predicted dim: 1 x #sample * split_ratio. Each element consists indices of predicted class.
        predicted = fit_model.predict(X_test)
        sum = 0
        for sample_index, class_predicted in enumerate(predicted):
            sum += y_test[sample_index, class_predicted]  # element will be 1 if correct and 0 if false
        scores.append(sum/y_test.shape[0])  # append the accuracy score

        # MSE
        deltaindex = np.where(deltas == fit_model.alpha_)[0]  # pick delta which is actually chosen
        MSE.append(fit_model.cv_values_[:, :, deltaindex])
    print("after train: ", time.process_time())
    return scores, MSE



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

    spike_times = spike_times[spike_times <= plot_time[1]]
    spike_times = spike_times[spike_times >= plot_time[0]]
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
    pylab.xticks(np.arange(arr_to_plot.shape[0]), ["M0", "M1", "M2", "M3"])
    pylab.ylabel(ylabel)
    pylab.title(title)
    pylab.savefig(filename, bbox_to_inches="tight")