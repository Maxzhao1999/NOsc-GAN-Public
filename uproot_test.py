# %%
# Saving the bins and counts from the root file for each histogram suing uproot
# testing here

import uproot
import numpy as np
import matplotlib.pyplot as plt

histograms = uproot.open('histograms.root')

saving = False

# bins and counts after loading have elements [th13,dcp,[bin]] and [th13,dcp,[counts]]
bins = []
counts = []
th13s = []
dcps = []


for i in range(len(histograms.keys())):

    th13s.append(float(str(histograms.keys()[i])[3:-3].split('_')[0]))
    dcps.append(float(str(histograms.keys()[i])[3:-3].split('_')[1]))

    bins.append(histograms[histograms.keys()[i]].edges)
    counts.append(histograms[histograms.keys()[i]].values)

if saving:
    np.save(
        f'bins_{min(th13s)}-{max(th13s)}-{len(np.unique(th13s))}_{min(dcps)}-{max(dcps)}-{len(np.unique(dcps))}.npy', bins)
    np.save(
        f'counts_{min(th13s)}-{max(th13s)}-{len(np.unique(th13s))}_{min(dcps)}-{max(dcps)}-{len(np.unique(dcps))}.npy', counts)

# %%
# Loading the bins and counts of each histogram and plotting them in matplotlib and saving them
# np.load()
bins = np.load(
    f'bins_{min(th13s)}-{max(th13s)}-{len(np.unique(th13s))}_{min(dcps)}-{max(dcps)}-{len(np.unique(dcps))}.npy')
counts = np.load(
    f'counts_{min(th13s)}-{max(th13s)}-{len(np.unique(th13s))}_{min(dcps)}-{max(dcps)}-{len(np.unique(dcps))}.npy')

for i in range(len(bins)):
    hist_label = r'$\theta_{13}$: ' + \
        f' {th13s[i]}\n' + r'$\delta_{cp}$: ' + f'{dcps[i]:.2f}'
    plt.hist(bins[i][:-1], bins[i], weights=counts[i], label=hist_label)
    plt.xlabel('Reconstructed Energy [GeV]', fontsize=15)
    plt.ylabel(r'$\nu_e$ appearance events', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(f'figures/hist_{th13s[i]}_{dcps[i]}.png')
    plt.legend(fontsize=12)
    plt.show()
