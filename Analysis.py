import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import datetime
import os
import cProfile
import pstats
import random
import time
import timeit
import pathlib
import shutil
import re
import io
import glob
import PIL
import PIL.Image
from matplotlib.colors import LogNorm
import matplotlib.cm as clm

# %%


def PoissonError(counts, prediction):

    pred_error = abs(counts - prediction)

    Full_Poisson_Error = []

    PE = []

    for i in range(len(counts)):

        for j in range(len(counts[0])):

            root_lambda = np.sqrt(counts[i, j])

            PE.append(root_lambda)

        Full_Poisson_Error.append(PE)

        PE = []

    error_diff = Full_Poisson_Error - pred_error

    return Full_Poisson_Error, pred_error, error_diff


def plot_D_predictions(list, D_real_pred, D_fake_pred, figure_loc):

    plt.figure(figsize=(12, 9))
    plt.plot(list, D_real_pred, label='D real predictions')
    plt.plot(list, D_fake_pred, label='D fake predictions')
    plt.xlabel('Epoch Number', fontsize=15)
    plt.ylabel('Discriminator prediction', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=12)
    plt.savefig(figure_loc + '/average_D_predictions.png')
    plt.show()
    plt.close()


def plot_errorgraph(list, full_pred_error_list, figure_loc):

    plt.figure(figsize=(12, 9))
    plt.plot(list, full_pred_error_list, label='Mean error of G')
    plt.xlabel('Epoch Number', fontsize=15)
    plt.ylabel('G mean prediction error', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=12)
    plt.savefig(figure_loc + '/average_G_prediction_error.png')
    plt.show()
    plt.close()


def multi_plot_errorgraph_all_th13s(list, multi_full_pred_error_list, figure_loc, full_all_th13s):

    plt.figure(figsize=(12, 9))
    for i in range(len(multi_full_pred_error_list)):
        plt.plot(
            list, multi_full_pred_error_list[i], label=f'{full_all_th13s[i]}')

    plt.xlabel('Epoch Number', fontsize=15)
    plt.ylabel('G mean prediction error', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=12)
    plt.savefig(figure_loc + '/all_th13s_average_G_prediction_error.png')
    plt.show()
    plt.close()


def multi_plot_errorgraph_all_dcps(list, multi_full_pred_error_list, figure_loc, full_all_dcps):

    plt.figure(figsize=(12, 9))
    for i in range(len(multi_full_pred_error_list)):
        plt.plot(
            list, multi_full_pred_error_list[i], label=f'{full_all_dcps[i]}')

    plt.xlabel('Epoch Number', fontsize=15)
    plt.ylabel('G mean prediction error', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=12)
    plt.savefig(figure_loc + '/all_dcps_average_G_prediction_error.png')
    plt.show()
    plt.close()


def plot_G_outputs(pred, index, error_diff, bins, counts, th13s, dcps, Full_Poisson_Error, num, figure_loc, close=True):
    if close:
        plt.close()
    plt.figure(figsize=(12, 9))

    if min(error_diff[index]) >= 0:
        plt.hist(bins[0][:-1], bins[0], weights=counts[index],
                 color='palegreen', label='real')
        plt.title(
            f'EPOCH {num}: for th13 = {th13s[index]} and dcp = {np.round(dcps[index], 3)}, INSIDE Poisson Error ', fontsize=15)
    else:
        plt.hist(bins[0][:-1], bins[0], weights=counts[index],
                 color='lightgrey', label='real')
        plt.title(
            f'EPOCH {num}: for th13 = {th13s[index]} and dcp = {np.round(dcps[index], 3)}, OUTSIDE Poisson Error ', fontsize=15)

    bincenters = 0.5 * (bins[0][1:] + bins[0][:-1])
    plt.errorbar(bincenters, counts[index], yerr=Full_Poisson_Error[index],
                 fmt='.', capsize=5, color='crimson',  label='Poisson Uncertainty')
    plt.hist(bins[0][:-1], bins[0], weights=pred,
             label='fake', histtype='step', color='blue')
    plt.xlabel('Reconstructed Energy [GeV]', fontsize=15)
    plt.ylabel(r'$\nu_e$ appearance events', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=12)

    plt.savefig(
        figure_loc + f'/generated_hist.{num}-{th13s[index]:{10}.{2}}-{dcps[index]:{10}.{2}}.png')


def error_spectrogram_plotter(X, Y, im, figure_loc, num):

    fig, ax = plt.subplots()

    params = {
        'axes.labelsize': 15,
        'font.size': 15,
        'font.family': 'sans-serif',  # Optionally change the font family to sans-serif
        'font.serif': 'Arial',  # Optionally change the font to Arial
        'legend.fontsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 15,
        # Using the golden ratio and standard column width of a journal
        'figure.figsize': [8.8, 8.8 / 1.618]
    }
    plt.rcParams.update(params)

    cm = ax.contourf(X, Y, im, 50, cmap=clm.plasma, origin=None)

    cb = fig.colorbar(cm)
    cb.ax.set_ylabel('Error')
    ax.set_xlabel('dcps')
    ax.set_ylabel('th13s')
    plt.savefig(figure_loc + f'/Error_spectrogram_epoch_{num}')
    plt.close()
