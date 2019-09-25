"""
Used to make a summary plot of network properties and classification accuracy
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from params import *

# parameters has to be tweaked manually
intra_mode = 'unimodal'
inter_mode = 'null'
PATH = os.getcwd() + '/data/sum/measures_intra={}_inter={}.p'.format(intra_mode, inter_mode)



"""
metric figures
"""

# measures = concat_files("measures*.p")
measures = pd.read_pickle(PATH)
metrics = measures.columns[2:]  # 4 different metrics, first two columns are indices
xticks = ["M0", "M1", "M2", "M3"]  # xlabels are module indices
ylabels = ["Pearson CC", "LvR", "spikes/sec", "Fano factor"]  # ylabels are units of metrics

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))

for measure_i in range(3):
    sns.lineplot(x="module index", y=metrics[measure_i], hue="network type", style="intra params", data=measures,
                 ci="sd", ax=axes[measure_i], legend=False)
    axes[measure_i].set(xticks=[0,1,2,3], xticklabels=xticks, ylabel=ylabels[measure_i], title=metrics[measure_i])
    # TODO: style parameter has to be changed manually

# not a part of for loop to make a legend
measure_i = 3
sns.lineplot(x="module index", y=metrics[measure_i], hue="network type", style="intra params", data=measures,
             ci="sd", ax=axes[measure_i])
axes[measure_i].set(xticks=[0,1,2,3], xticklabels=xticks, ylabel=ylabels[measure_i], title=metrics[measure_i])

# handling legend in a dirty way
handles, labels = axes[-1].get_legend_handles_labels()
axes[-1].legend(handles=handles[1:4]+handles[5:], labels=labels[1:4]+labels[5:], bbox_to_anchor=(1.05, 1.1))

# save the figure
fig.tight_layout()
plt.savefig("ultimate_intra={}_inter={}.pdf".format(intra_mode, inter_mode), bbox_to_inches="tight")



# """
# training figures
# """
# accuracy_train = concat_files("accuracy_*.p")
# MSE_train = concat_files("MSE_*.p")
#
# ylabels = ["Accuracy", "MSE"]
# datas = [accuracy_train, MSE_train]
# fig, axes = plt.subplots (nrows=2, ncols=1, figsize=(6,8))
# for measure_i in range(2):
#     sns.barplot(x="module index", y="value", hue="network type", data=datas[measure_i], ci="sd",
#                 ax=axes[measure_i])
#     axes[measure_i].set(ylabel=ylabels[measure_i], xticklabels=xticks)
# fig.tight_layout()
# plt.savefig("training.pdf", bbox_to_inches="tight")
