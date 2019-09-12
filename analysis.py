####
# script to compute useful measures such as correlation, firing rate and to plot the result
####

import sys
import os
import glob
import time
import itertools
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split

from params import *
#from plotter import plot_raster

# rsrc = resource.RLIMIT_AS
# soft, hard = resource.getrlimit(rsrc)
# print("Limit starts as:", soft, hard)
# resource.setrlimit(rsrc, (7168, 7200))
# soft, hard = resource.getrlimit(rsrc)
# print("Limit is now:", soft, hard)

# depend on the machine, CHECK BEFOREHAND
PATH = os.getcwd() + "/data/"
# PATH = os.environ["TMPDIR"] + "/"
print("path: ", PATH)



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
            indices = np.where(neuron == np.array(spike_senders))[0]  # [12, 17, 21,...] indices of spike time of a current neuron
            st = spike_times[indices] - 0.00001  # [0.9999, 18.999, 238.9999...] # dirty trick to make binning easier
            boolean_arr[nid, np.int_(st//bin_size)] = True  # now the array is full with binned spike train
        sum_pearson_coef += np.corrcoef(boolean_arr)[0,1]  # compute sum of Pearson corr. coef.
    return sum_pearson_coef / num_pair


# II. LvR
def revised_local_variation(spike_times, spike_senders):
    """
    compute the revised_local_variation (LvR) suggested by Shinomoto, 2009. neuron-wise and return the avg. value
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
    hist = hist * 1000 / bin_size
    return np.var(hist) / np.mean(hist)


# def fano_factor_fr(spike_times, record_time):
#     bin_size = 10  # width of a single bin in ms
#     bins = np.arange(0, record_time + 0.1, bin_size)  # define bin edges
#     hist, edges = np.histogram(spike_times, bins=bins)
#     hist = hist * 1000 / bin_size
#     # compute firing rate using Gaussian kernel on top of histogram and save value for every tick (ms)
#     tick = 0.1
#     firingrate = gaussian_kde(dataset=hist).evaluate(np.arange(0, record_time+tick, tick))
#     return np.var(firingrate) / np.mean(firingrate)


# V. classification
def train(volt_values, target_output, split_ratio=0.2):
    """
    function to train a simple linear regression to fit the snapshot of membrane potential to binary classification
    using a ridge regression with cross-validation for regularization parameter.
    :param volt_values: np.arr, shape: len.of stim. presentation x N_E.
    snapshots of membrane potential at each stimuli offset.
    :param target_output: np.arr, shape: num. of stimuli x len. of stim. presentation. @sym_seq in the main.py
    :param split_ratio: float, percentage of the data to be used for the test
    :return: list, saves the score for each module
    """
    scores = np.zeros(module_depth) # array to save accuracy score for each module
    MSE = np.zeros(module_depth)
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
        scores[mod_i] = sum / y_test.shape[0]  # append the accuracy score

        # MSE
        deltaindex = np.where(deltas == fit_model.alpha_)[0]  # pick delta which is actually chosen
        MSE[mod_i] = np.mean(fit_model.cv_values_[:, :, deltaindex], axis=(0,1,2))
    return scores, MSE



######################################################################
# the real script starts here
######################################################################

"""
parse the parameters and init. data structures
"""
runnum = 5  # default num. of repetition is 5
mode = sys.argv[1]  # input should be either "noise", "random" or "topo"
sim_time = eval(sys.argv[2])

measures = np.zeros((runnum, module_depth, 4))
accuracy_train = np.zeros((runnum, module_depth))
MSE_train = np.zeros((runnum, module_depth))



"""
load all raw data in appropriate formats
"""
def load_data(filename):
    """
    load all the data with a given filename inside a PATH in a single list
    :param filename: str, names of the files to load. Using RegEx is recommended.
    :return: list
    """
    all_files = glob.glob(os.path.join(PATH, filename))
    myl = []
    for f in all_files:
        myl.append(np.loadtxt(f))  # load all files from different kernels to one data structure
    return myl


reshape_arr = lambda flat: np.reshape(flat, (-1, module_depth, N_E))  # reshape the array in (time,module,neurons)

for runindex in range(runnum):
    # membrane potential
    if (mode=="random") or (mode=="topo"):
        expression = "volt_{}_run={}-*.dat".format(mode, runindex)
        volt_values = reshape_arr(np.hstack(np.array(load_data(expression))))

    # spikes
    spike_times = []
    spike_senders = []
    for mod_i in range(module_depth):
        expression = "spike_{}_run={}-4000{}-*.dat".format(mode, runindex, mod_i + 1)
        spike_arr = np.vstack(np.array(load_data(expression)))
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
                                        fano_factor(spike_times=times, record_time=sim_time)
                                        ]
    print("measures are computed now: ", time.process_time())



    """
    train the classifier and test it
    """
    if (mode=="random") or (mode=="topo"):
        accuracy_train[runindex], MSE_train[runindex] = train(volt_values=volt_values,
                                  target_output=np.loadtxt(PATH + "stimuli_{}_run={}".format(mode, runindex)))
        print("training is done: ", time.process_time())



"""
save the summary data to the file
"""
def reformat_df(nparr):
    """
    transform 2-d np.array to a data frame and collapse them
    :param nparr: np.array
    :return: pd.df
    """
    df_new = pd.melt((pd.DataFrame(nparr, dtype=float)), var_name="module index")
    df_new["network type"] = mode
    return df_new


m,n,r = measures.shape
out_arr = np.column_stack((np.tile(np.arange(n),m), measures.reshape(m*n, -1)))
df_measure = pd.DataFrame(out_arr,
                          columns=["module index", "synchrony", "irregularity", "firing rate", "variability"])
df_measure["network type"] = mode
df_acc = reformat_df(accuracy_train)
df_mse = reformat_df(MSE_train)

df_measure.to_pickle("measures_{}.p".format(mode))  # save the data
df_acc.to_pickle("accuracy_train_{}.p".format(mode))
df_mse.to_pickle("MSE_train_{}.p".format(mode))



"""
raster plot here because I don't want to load spike data in plot.py
"""
from plotter import plot_raster

mod_i = 2  # which layer to plot (M2)
plot_raster(filename="raster_{}.pdf".format(mode),
                     spike_times=spike_times[mod_i], spike_senders=spike_senders[mod_i], layer=mod_i,
                     num_to_plot=500)
#
# """
# raster plot here because I don't want to load spike data in plot.py
# """
# mod_i = 2  # which layer to plot (M2)
# plot_raster(filename="raster_{}.pdf".format(mode),
#                      spike_times=spike_times[mod_i], spike_senders=spike_senders[mod_i], layer=mod_i,
#                      num_to_plot=500)
