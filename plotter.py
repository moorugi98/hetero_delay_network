import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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


def plot_raster(filename, spike_times, spike_senders, layer, num_to_plot = 100, plot_time = [6000, 8000]):
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
    spike_times = spike_times[spike_times <= plot_time[1]]
    spike_times = spike_times[spike_times >= plot_time[0]]
    rand_choice = np.random.randint(0 + N*layer, N*(layer+1), num_to_plot) # choose neurons to plot randomly
    mask = np.isin(spike_senders, rand_choice)
    a0.scatter(spike_times[mask], spike_senders[mask], s=0.1, c="r")
    a1.hist(spike_times, bins=5.0)
    plt.savefig(filename, bbox_to_inches="tight")



def plot_result(filename, arr_to_plot, title, ylabel):
    """
    helper function to plot the result of various measures
    :filename: str, name of the file to save
    :param arr_to_plot: np.arr, the data to plot
    :param title: str, the title of the plot
    :param ylabel: str, ylabel of the plot
    """
    plt.figure()
    plt.plot(arr_to_plot)
    plt.xticks(np.arange(arr_to_plot.shape[0]), ["M0", "M1", "M2", "M3"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename, bbox_to_inches="tight")





######################

"""
load the data
"""
def concat_files(expression):
    PATH = os.getcwd()+ "/data"
    all_files = glob.glob(os.path.join(PATH, expression))
    df_from_each_file = (pd.read_pickle(f) for f in all_files)  # use generator to give out data frame each time
    return pd.concat(df_from_each_file, ignore_index=False)

measures = concat_files("measures*.p")
accuracy_train = concat_files("accuracy_*.p")
MSE_train = concat_files("MSE_*.p")


print(measures["firing rate"])
"""
metric figures
"""

titles = measures.columns  # titles of figures
ylabels = ["Pearson CC", "LvR", "spikes/sec", "Fano factor"]  # ylabels

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(24, 4))
for measure_i in range(4):
    sns.lineplot(x="module index", y=titles[measure_i+1], hue="network type", data=measures,
                 ci="sd", ax=axes[measure_i])
    axes[measure_i].set(xticklabels=[None, "M0", "M1", "M2", "M3"])
fig.tight_layout()
plt.savefig("ultimate.pdf", bbox_to_inches="tight")



"""
training figures
"""
ylabels = ["Accuracy", "MSE"]
datas = [accuracy_train, MSE_train]
fig, axes = plt.subplots (nrows=2, ncols=1, figsize=(6,10))
for measure_i in range(2):
    sns.barplot(x="module index", y="value", hue="network type", data=datas[measure_i], ci="sd",
                ax=axes[measure_i])
    axes[measure_i].set(ylabel=ylabels[measure_i], xticklabels=["M0", "M1", "M2", "M3"])
fig.tight_layout()
plt.savefig("training.pdf", bbox_to_inches="tight")
