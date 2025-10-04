# Run this with some .pkls in analyse-pkls
# Currently set up to compare a model with lambda=0 everywhere (null model) and model with SSL
# Will calculate a one-tailed p-value on the improvement in mean best Dice score

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import os
import pickle
import scipy

ANALYSE_PATH = 'analyse-pkls'

def plot_dice(ax: Axes, dices, num_epochs, label=None, c=None, fmt='.-'):
    # x = np.array(range(len(dices)))
    ax.plot(np.linspace(0,num_epochs,len(dices)), dices, fmt, label=label, c=c)
    ax.set_title("Validation Dice During Training")
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

max_null = []
max_ssl = []

def max_n(arr, n):
    return np.sort(arr)[-n:]

for path in os.listdir(ANALYSE_PATH):
    print(path)
    with open(os.path.join(ANALYSE_PATH, path), 'rb') as file:
        data = pickle.load(file)

    col_t = data['lambdas'].mean() * 75
    num_epochs = len(data['lambdas'])
    fmt = '--' if col_t < 1e-6 else '-'
    plot_dice(dice_plot, data['dice'], num_epochs, c=cmap(col_t), label=path, fmt=fmt)
    plot_losses(losses_plot, data['sls'], data['usls'], label=path, c_sup=cmap(col_t), c_unsup = cmap(col_t + 0.2))
    plot_lambdas(lambdas_plot, data['lambdas'], label=path, c=cmap(0.))

    # null will have col_t = 0 (maybe floating point error)
    (max_null if col_t < 1e-5 else max_ssl).append(max_n(data['dice'], 1))

max_null = np.array(max_null).flatten()
max_ssl = np.array(max_ssl).flatten()
print(f"top nulls (concat\'d): {max_null}\ntop ssls (concat\'d): {max_ssl}")

best_null = (np.mean(max_null), np.std(max_null, ddof=1)/np.sqrt(len(max_null)))
best_ssl = (np.mean(max_ssl), np.std(max_ssl, ddof=1)/np.sqrt(len(max_ssl)))

z_score = (best_ssl[0] - best_null[0]) / np.sqrt(best_null[1]**2 + best_ssl[1]**2)
p_value = scipy.stats.norm.sf(abs(z_score))
print(f"mean best null: {best_null[0]:.03f} +/- {best_null[1]:.03f}\nmean best ssl: {best_ssl[0]:.03f} +/- {best_ssl[1]:.03f}\np-value: {p_value:.010f} ({z_score:.02f}) sigma")


ssl_handle, = dice_plot.plot([], [], '-', c=cmap(0.), label='With SSL')
null_handle, = dice_plot.plot([], [], '--', c=cmap(0.), label='Without SSL')
dice_plot.legend(handles=[ssl_handle, null_handle])
# dice_plot.legend()
# dice_plot.set_ylim((0.75, 0.85))

ssl_sl_handle, = dice_plot.plot([], c=cmap(0.75), label='Supervised Loss (SSL)')
ssl_usl_handle, = dice_plot.plot([], c=cmap(0.95), label='Unsupervised Loss (SSL)')
null_sl_handle, = dice_plot.plot([], c=cmap(0.0), label='Supervised Loss (No SSL)')
null_usl_handle, = dice_plot.plot([], c=cmap(0.2), label='Unsupervised Loss (No SSL)')
losses_plot.legend(handles=[ssl_sl_handle, ssl_usl_handle, null_sl_handle, null_usl_handle])
losses_plot.set_ylim((0., 0.2))

x = np.linspace(0.6, 0.9, 256)
mean_dist_plot.plot(x, scipy.stats.norm.pdf(x, loc=best_null[0], scale=best_null[1]), label='best nulls', c=cmap(0.0))
mean_dist_plot.plot(x, scipy.stats.norm.pdf(x, loc=best_ssl[0], scale=best_ssl[1]), label='best ssls', c=cmap(0.5))
mean_dist_plot.set_xlabel("Best Dice Score")
mean_dist_plot.legend()

plt.show()
