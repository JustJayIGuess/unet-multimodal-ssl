# Usage: analyse-pkls.py <path-to-folder-containing-logs>
# e.g., analyse-pkls.py training-logs

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import os
import pickle
import scipy
import sys
from tqdm import tqdm

ANALYSE_PATH = sys.argv[1]

def plot_dice(ax: Axes, dices, num_epochs, label=None, c=None, fmt='.-'):
    # x = np.array(range(len(dices)))
    ax.plot(np.linspace(0,num_epochs,len(dices)), dices, fmt, label=label, c=c)
    ax.set_title("Validation Dice During Training - 6 Labelled Volumes")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Dice Score")

def plot_losses(ax: Axes, sls, usls, label='', c_sup=None, c_unsup=None):
    ax.plot(usls, label=('Unsupervised Loss ' + label), c=c_unsup)
    ax.plot(sls, label=('Supervised Loss ' + label), c=c_sup)
    ax.set_title("Losses")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    
def plot_lambdas(ax: Axes, lambdas, label=None, c=None):
    ax.plot(lambdas, label=label)
    ax.set_title("SSL $\\lambda$ Values")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("$\\lambda$")

dice_plot = plt.figure().add_subplot()
losses_plot = plt.figure().add_subplot()
lambdas_plot = plt.figure().add_subplot()
mean_dist_plot = plt.figure().add_subplot()
cmap = plt.colormaps['viridis']

max_nulls = []
max_ssls = []

def max_n(arr, n):
    return np.sort(arr)[-n:]


for log_path in tqdm(os.listdir(ANALYSE_PATH)):
    parent_path = os.path.join(ANALYSE_PATH, log_path)
    for filename in os.listdir(parent_path):
        if not (".pkl" in filename):
            continue
                
        path = os.path.join(parent_path, filename)
        with open(path, 'rb') as file:
            data = pickle.load(file)
        
        portion_labelled = data['dataset_sizes']['labelled'] / (data['dataset_sizes']['unlabelled'] + data['dataset_sizes']['labelled'])

        col_t = data['lambdas'].mean() * 40
        num_epochs = len(data['lambdas'])
        fmt = '-' if col_t < 1e-6 else '-'
        plot_dice(dice_plot, data['dice'], num_epochs, c=cmap(col_t), label=filename, fmt=fmt)
        plot_losses(losses_plot, data['sls'], data['usls'], label=filename, c_sup=cmap(col_t), c_unsup = cmap(col_t + 0.2))
        plot_lambdas(lambdas_plot, data['lambdas'], label=filename, c=cmap(0.))

        # null will have col_t = 0 (maybe floating point error)
        (max_nulls if col_t < 1e-5 else max_ssls).append(max_n(data['dice'], 1))

best_dices_ssl = []
best_dices_null = []

max_ssl = np.array(max_ssls).flatten()
max_null = np.array(max_nulls).flatten()
best_null = (np.mean(max_null), np.std(max_null, ddof=1)/np.sqrt(len(max_null)))
best_ssl = (np.mean(max_ssl), np.std(max_ssl, ddof=1)/np.sqrt(len(max_ssl)))

z_score = (best_ssl[0] - best_null[0]) / np.sqrt(best_null[1]**2 + best_ssl[1]**2)
p_value = scipy.stats.norm.sf(abs(z_score))
print(f"mean best null: {best_null[0]:.03f} +/- {best_null[1]:.03f}\nmean best ssl: {best_ssl[0]:.03f} +/- {best_ssl[1]:.03f}\np-value: {p_value:.010f} ({z_score:.02f}) sigma")

x = np.linspace(0.6, 1.0, 256)
null_norm = max_null.size
ssl_norm = max_ssl.size
mean_dist_plot.plot(x, scipy.stats.norm.pdf(x, loc=best_null[0], scale=best_null[1]), label='Without SSL', c=cmap(0.0))
mean_dist_plot.plot(x, scipy.stats.norm.pdf(x, loc=best_ssl[0], scale=best_ssl[1]), label='With SSL', c=cmap(0.65))
mean_dist_plot.set_xlabel("Best Dice Score")
mean_dist_plot.set_ylabel("Probability Density")
mean_dist_plot.set_title("Distribution of Best Dice Scores - 6 Labelled Volumes")
mean_dist_plot.legend()

plt.show()
