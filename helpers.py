"""
Define all the necessary functions used here to avoid scripts getting messy
"""


import glob
import itertools
import random

import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split

from params import *


# I. synchrony
def pairwise_corr(spike_times, spike_senders, record_time):
    """
    compute the avg. Pearson pairwise_lscorr for randomly selected pair of spike trains
    : param spike_times: list, nest.GetStatus(spike_detector, ["event"])[0]["times"]
    :param spike_senders: list, nest.GetStatus(spike_detector, ["event"])[0]["senders"]
    : param record_time: int, recording time in ms
    :return: int, avg. Pearson pairwise_corr
    """
    # parameter
    sum_pearson_coef = 0
    num_pair = 500  # num. of pair to compute avg. pairwise_cor
    bin_size = 2  # bin_size for computing spike train

    spike_times = np.array(spike_times) - t_onset  # control for t_onset ruining time bin
    spike_senders = np.array(spike_senders)
    pairs = random.sample(list(itertools.combinations(np.unique(spike_senders), 2)), num_pair)  # num_pair rand. pairs

    for pair in pairs:
        boolean_arr = np.zeros((2, int(record_time // bin_size)), dtype=bool)  # init spike train
        for nid, neuron in enumerate(pair):  # iterate over two neurons in each pair
            indices = np.where(neuron == np.array(spike_senders))[0]  # indices of spike time of a current neuron
            st = spike_times[indices] - 0.00001  # [0.9999, 18.999, 238.9999...] # dirty trick to make binning easier
            boolean_arr[nid, np.int_(st//bin_size)] = True  # now the array is full with binned spike train
        sum_pearson_coef += np.corrcoef(boolean_arr)[0,1]  # compute sum of Pearson corr. coef.
    return sum_pearson_coef / num_pair


# II. LvR
def revised_local_variation(spike_times, spike_senders):
    """
    compute the revised_local_variation (LvR) suggested by (Shinomoto, 2009) neuron-wise and return the avg. value
    :param spike_times: list, nest.GetStatus(spike_detector, ["event"])[0]["times"]
    :param spike_senders: list, nest.GetStatus(spike_detector, ["event"])[0]["senders"]
    :return: int, mean LvR value
    """
    spike_senders = np.array(spike_senders)
    neuron_list = np.unique(spike_senders)  # all unique gids of neurons
    lvr = np.zeros(neuron_list.shape[0])  # save lvr for each neuron to this array

    for ni, neuron in enumerate(neuron_list):
        indices = np.where(neuron == spike_senders)[0]
        isi = np.ediff1d(np.sort(np.array(spike_times)[indices]))  # inter spike interval
        if isi.shape[0] < 2:
            lvr[ni] = np.nan
        else:
            lvr[ni] = ((3 / (isi.shape[0]-1)) *
                       np.sum(
                       (1 - 4*isi[:-1]*isi[1:] / (isi[:-1] + isi[1:])**2) *
                       (1 + (4*tau_ref) / (isi[:-1] + isi[1:]))))
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
    return len(spike_senders) / (record_time * N)


# IV. Fano factor
def fano_factor(spike_times, record_time, N):
    """
    compute the Fano factor by using the population avg. firing rate with 10 ms bin.
    :param spike_times: list, nest.GetStatus(spike_detector, ["event"])[0]["times"]
    :param record_time: int, recording time in ms
    :param N: int, number of neurons recorded
    :return: Fano factor
    """
    bin_size = 10  # width of a  single bin in ms
    bins = np.arange(0, record_time+0.1, bin_size)  # define bin edges
    hist, edges = np.histogram(spike_times, bins=bins)
    normed = (hist/N) * (1000/bin_size)  # num of spike of a single neuron per one sec
    return np.var(normed) / np.mean(normed)


def train(volt_values, target_output, split_ratio=0.2):
    """
    function to train a simple linear regression to fit the snapshot of membrane potential (state matrix) to binary classification
    using a ridge regression with cross-validation for regularization parameter.
    :param volt_values: np.arr, shape: len.of stim. presentation x N_E.
    snapshots of membrane potential at each stimuli offset.
    :param target_output: np.arr, shape: num. of stimuli x len. of stim. presentation. @sym_seq in the main.py
    :param split_ratio: float, percentage of the data to be used for the test
    :return: list, saves the score for each module
    """
    scores = np.zeros(module_depth)  # array to save accuracy score for each module
    MSE = np.zeros(module_depth)
    for mod_i in range(module_depth):
        # split the data into training and test sets
        # x_train dim: #train_sample(#screenshots) x #features(#neurons)
        # y_train dim: #train_sample * #classes(stimuli)
        print(np.transpose(np.int_(target_output)).shape)
        x_train, x_test, y_train, y_test = train_test_split(np.transpose(volt_values[mod_i, :]),  # for each module
                                                            np.transpose(np.int_(target_output)), test_size=split_ratio)

        # linear ridge regression with cross-validation for regularization parameter
        # deltas = [0.01, 0.1, 1, 10, 100]  # regularization parameter
        deltas = [1e0, 1e3, 1e4, 2e4, 5e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
        fit_model = lm.RidgeClassifierCV(alphas=deltas, fit_intercept=True, store_cv_values=True)\
            .fit(X=x_train, y=y_train)

        # use the trained weight to predict the class of @y_test. Use WTA operation, without giving confidence level
        # predicted dim: 1 x #test sample. Each element consists indices of predicted class.
        predicted = fit_model.predict(x_test) # dim: sample num x 1. Each entry indicates that n-th class is predicted.
        sum = 0  # count how many samples of y_test gets classified correctly
        for sample_index, class_predicted in enumerate(predicted):
            sum += y_test[sample_index, class_predicted]  # entry of y_test are 0 and 1
        scores[mod_i] = sum / y_test.shape[0]  # normalize to 1 and save the accuracy for each module
        # print("weights: ", fit_model.coef_[:4, :10])
        # print("intercepts: ", fit_model.intercept_[:10])
        print("reg.params.: ", fit_model.alpha_) # shit doesn't work

        # MSE
        deltaindex = np.where(deltas == fit_model.alpha_)[0]  # pick delta which is actually chosen
        MSE[mod_i] = np.mean(fit_model.cv_values_[:, :, deltaindex], axis=(0,1,2))  # average over all samples & feats.
    return scores, MSE


# def load_data(PATH, filename):
#     """
#     load all the data with a given filename inside a PATH in a single list
#     :param filename: str, names of the files to load. Using RegEx is recommended.
#     :return: list
#     """
#     all_files = glob.glob(os.path.join(PATH, filename))
#     # myl = []
#     # for f in all_files:
#     #     myl.append(np.loadtxt(f))  # load all files from different kernels to one data structure
#     # return myl
#     arrays = [np.loadtxt(f) for f in all_files]
#     return np.concatenate(arrays, axis=0)


def reshape_arr(flat):
    return np.reshape(flat, (-1, module_depth, N_E))  # reshape the array in (time,module,neurons)


def reformat_df(network_mode, nparr):
    """
    transform 2-d np.array to a data frame and collapse them
    :param nparr: np.array
    :return: pd.df
    """
    df_new = pd.melt((pd.DataFrame(nparr, dtype=float)), var_name="module index")
    df_new["network type"] = network_mode
    'network type', 'intra type', 'inter type',
    'intra params', 'inter params'
    return df_new


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
    fig, axes = plt.subplots(nrows=num_to_plot+1, ncols=1)
    for neuron_index in range(num_to_plot):
        axes[neuron_index].plot(times, voltages[:,neuron_index], "gray")
    axes[-1].plot(times, np.mean(voltages, axis=1), "blue") # plot the avg.
    plt.xlabel("time(ms)", fontsize=30)
    plt.ylabel("potential(mV)", fontsize=30)
    plt.savefig(filename, bbox_to_inches="tight")


# def plot_result(filename, arr_to_plot, title, ylabel):
#     """
#     helper function to plot the result of various measures
#     :filename: str, name of the file to save
#     :param arr_to_plot: np.arr, the data to plot
#     :param title: str, the title of the plot
#     :param ylabel: str, ylabel of the plot
#     """
#     plt.figure()
#     plt.plot(arr_to_plot)
#     plt.xticks(np.arange(arr_to_plot.shape[0]), ["M0", "M1", "M2", "M3"])
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.savefig(filename, bbox_to_inches="tight")