####### DEFINE ALL FUNCTIONS ######
import uproot
from PIL import Image
from os import listdir
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
# %%


def merge_dataset(root_name, saving=False, plotting=False):

    bins = []
    counts = []
    th13s = []
    dcps = []

    s = time.time()

    for i in range(len(root_name)):
        histograms = uproot.open(
            f'{root_name[i]}.root')

        for j in histograms.keys():

            th13s.append(float(str(j)[3:-3].split('_')[0]))
            dcps.append(float(str(j)[3:-3].split('_')[1]))
            bins.append(histograms[j].edges)
            counts.append(histograms[j].values)

        e = time.time()

        print('time', e - s)

    if saving:
        np.save(
            f'bins_{min(th13s)}-{max(th13s)}-{len(np.unique(th13s))}_{min(dcps)}-{max(dcps)}-{len(np.unique(dcps))}.npy', bins)
        np.save(
            f'counts_{min(th13s)}-{max(th13s)}-{len(np.unique(th13s))}_{min(dcps)}-{max(dcps)}-{len(np.unique(dcps))}.npy', counts)

    bins = np.load(
        f'bins_{min(th13s)}-{max(th13s)}-{len(np.unique(th13s))}_{min(dcps)}-{max(dcps)}-{len(np.unique(dcps))}.npy')
    counts = np.load(
        f'counts_{min(th13s)}-{max(th13s)}-{len(np.unique(th13s))}_{min(dcps)}-{max(dcps)}-{len(np.unique(dcps))}.npy')

    if plotting:
        for i in range(5):
            hist_label = r'$\theta_{13}$: ' + \
                f' {th13s[i]}\n' + r'$\delta_{cp}$: ' + f'{dcps[i]:.2f}'
            plt.hist(bins[i][:-1], bins[i],
                     weights=counts[i], label=hist_label)
            plt.xlabel('Reconstructed Energy [GeV]', fontsize=15)
            plt.ylabel(r'$\nu_e$ appearance events', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(fontsize=12)
            plt.savefig(f'figures/hist_{th13s[i]}_{dcps[i]}.png')
            plt.show()

    return bins, np.float32(counts), th13s, dcps


# bins, counts, th13s, dcps = merge_dataset(root_name=[
#                                           'histograms102400a', 'histograms102400b'], saving=True, plotting=False)
#
# norm_th13s = MinMaxScaler(feature_range=(-1, 1))
# norm_dcps = MinMaxScaler(feature_range=(-1, 1))
# #
# # normalising the dcps [0,1] to improve loss calculations and training
# th13s = np.array(th13s)
# th13s = th13s.reshape(len(th13s), 1)
# norm_th13s.fit(th13s)
# th13s = norm_th13s.transform(th13s)
# th13s = th13s.reshape(th13s.shape[0])
# th13s = th13s.tolist()
# #
# dcps = np.array(dcps)
# dcps = dcps.reshape(len(dcps), 1)
# norm_dcps.fit(dcps)
# dcps = norm_dcps.transform(dcps)
# dcps = dcps.reshape(dcps.shape[0])
# dcps = dcps.tolist()
#
# np.save('th13s_sindcp_normed', th13s)
# np.save('dcps_sindcp_normed', dcps)
# #np.save('bins_sindcp.npy', bins)
# #np.save('counts_sindcp.npy', counts)
#
#
# bins, counts, th13s, dcps = merge_dataset(root_name=[
#                                           'histograms_between_a', 'histograms_between_b'], saving=True, plotting=False)
#
# # normalising the dcps [0,1] to improve loss calculations and training
# th13s = np.array(th13s)
# th13s = th13s.reshape(len(th13s), 1)
# th13s = norm_th13s.transform(th13s)
# th13s = th13s.reshape(th13s.shape[0])
# th13s = th13s.tolist()
# #
# dcps = np.array(dcps)
# dcps = dcps.reshape(len(dcps), 1)
# dcps = norm_dcps.transform(dcps)
# dcps = dcps.reshape(dcps.shape[0])
# dcps = dcps.tolist()
#
# np.save('th13s_between_normed', th13s)
# np.save('dcps_between_normed', dcps)
# #np.save('bins_between.npy', bins)
# #np.save('counts_between.npy', counts)
