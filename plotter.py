"""
Used to make a summary plot of network properties and classification accuracy
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from params import *

# parameters has to be tweaked manually
intra_mode = 'unimodal'
inter_mode = 'unimodal'
PATH = os.getcwd() + '/data/sum/'
xticks = ["M0", "M1", "M2", "M3"]  # xlabels are module indices



"""
measures figures
"""
# load data and melt in an appropriate format
measures = pd.read_csv(PATH + 'measures_intra={}_inter={}.csv'.format(intra_mode, inter_mode), keep_default_na=False)
metrics = measures.columns[1:]  # 4 different metrics, first column is indices
measures = measures
measures['network type'] = pd.Categorical(measures['network type'], categories=['noise', 'random', 'topo'])
measures = measures.melt(id_vars=['module index', 'network type', 'intra type', 'intra params', 'inter type',
                                  'inter params'], var_name='metric').sort_values(by=['network type', 'module index'])
print(measures)

# plot
sns.set(font_scale=1.5)
g = sns.FacetGrid(measures, col="metric", row="network type", hue='network type',
                  sharex=True, sharey='col', margin_titles=False)
g.map_dataframe(sns.lineplot, "module index", 'value', style='intra params', legend='full')

# ticks and labels
ylabels = ["Pearson CC", "LvR", "spikes/sec", "Fano factor"]  # ylabels are units of metrics
for row_i in range(3):
    for ax_i, ax in enumerate(g.axes[row_i]):
        ax.set(title=None, ylabel=None)
for ax_i, ax in enumerate(g.axes[0]):
    ax.set(title=metrics[ax_i], ylabel=ylabels[ax_i], xticklabels=xticks, xticks=np.arange(4))

# legends
g.add_legend()
handles, labels = g.axes[-1][-1].get_legend_handles_labels()
g.axes[-1][-1].legend(handles=handles[5:], labels=labels[5:], bbox_to_anchor=(1.9, 1.0))

# save the figure
plt.savefig("ultimate_intra={}_inter={}.pdf".format(intra_mode, inter_mode), bbox_to_inches="tight")



# """
# training figures
# """
# training = pd.read_csv(PATH + 'training_intra={}_inter={}.csv'.format(intra_mode, inter_mode), keep_default_na=False)
# training = training.melt(id_vars=['module index', 'network type', 'intra type',
#        'inter type', 'intra params', 'inter params'], var_name='metric')
# print(training)
#
# sns.set(font_scale=2)
# g = sns.catplot(x="module index", y="value", hue='intra params', data=training,
#             kind='bar', row="metric", col='network type', sharey='row', margin_titles=True,
#             ci='sd', alpha=0.7)
# for ax in g.axes[0]:
#     ax.axhline(y=0.1, color='black', linewidth=2.0)
# # g = sns.catplot(x="module index", y="value", hue='network type', data=training,
# #             kind='bar', col='metric', sharey=False, margin_titles=True,
# #             ci='sd', alpha=0.7)
# # g.fig.set_size_inches(15,8)
#
# plt.savefig("training_intra={}_inter={}.pdf".format(intra_mode, inter_mode), bbox_to_inches="tight")
