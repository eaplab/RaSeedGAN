# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 16:42:24 2021
@author: guemesturb
"""


import os
import re
import cmasher as cmr
import matplotlib
import numpy as np
import TheArtist
import matplotlib.pyplot as plt


def main():

    plt.rc('text', usetex='True')
    plt.rc('font', family='Serif', size='6')
    # plt.rc('axes', titlesize='6')

    fig_width_pt = 510
    inches_per_pt = 1.0 / 72.27  
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width * 0.23 

    figure = plt.figure('panel01', figsize=(fig_width, fig_height))
    im = []

    ax03 = figure.add_axes([0.01, 0.750, 0.1, 0.05])
    ax03.text(0.2, 0.5, 'Test Case 3 -- NOAA Sea Surface Temperature', fontsize=8)
    ax03.axis("off")

    ax05 = figure.add_axes([0.040, 0.90, 0.21, 0.04])
    ax06 = figure.add_axes([0.260, 0.90, 0.21, 0.04])
    ax07 = figure.add_axes([0.480, 0.90, 0.21, 0.04])
    ax08 = figure.add_axes([0.700, 0.90, 0.21, 0.04])
    ax05.text(0.5, 0.5, 'LR Input',              horizontalalignment='center', verticalalignment='center', transform=ax05.transAxes, fontsize=8)
    ax06.text(0.5, 0.5, 'Sparse HR Reference',   horizontalalignment='center', verticalalignment='center', transform=ax06.transAxes, fontsize=8)
    ax07.text(0.5, 0.5, 'RaSeedGAN',    horizontalalignment='center', verticalalignment='center', transform=ax07.transAxes, fontsize=8)
    ax08.text(0.5, 0.5, 'Complete HR Reference', horizontalalignment='center', verticalalignment='center', transform=ax08.transAxes, fontsize=8)
    ax05.axis("off")
    ax06.axis("off")
    ax07.axis("off")
    ax08.axis("off")

    """
        SST
    """

    ax29 = figure.add_axes([0.040, 0.05, 0.210, 0.85])
    ax30 = figure.add_axes([0.260, 0.05, 0.210, 0.85])
    ax31 = figure.add_axes([0.480, 0.05, 0.210, 0.85])
    ax32 = figure.add_axes([0.700, 0.05, 0.210, 0.85])

    filename = f"/storage/aguemes/gan-piv/sst/ss08/results/predictions_architecture-01-noise-000.npz"
    data = np.load(filename)
    dns_target = data['dns_target'] 
    hr_predic  = data['hr_predic'] 
    hr_target  = data['hr_target'] 
    lr_target  = data['lr_target'] 
    fl_target  = data['fl_target']
    hr_target[fl_target == 0] = np.nan

    cmap_t = plt.get_cmap('cmr.guppy_r').copy()
    cmap_t.set_bad(color='w')

    im.append(ax29.imshow(lr_target[0,:,:,0], origin="lower", extent=[0, 360, -90, 90], cmap=cmap_t, vmin=-5, vmax=30))
    im.append(ax30.imshow(hr_target[0,:,:,0], origin="lower", extent=[0, 360, -90, 90], cmap=cmap_t, vmin=-5, vmax=30))
    im.append(ax31.imshow(hr_predic[0,:,:,0], origin="lower", extent=[0, 360, -90, 90], cmap=cmap_t, vmin=-5, vmax=30))
    im.append(ax32.imshow(dns_target[0,:,:,0], origin="lower", extent=[0, 360, -90, 90], cmap=cmap_t, vmin=-5, vmax=30))

    axins = ax30.inset_axes([0.01, 0.1, 0.8, 0.8])
    axins.imshow(hr_target[0,:,:,0], origin="lower", extent=[0, 360, -90, 90], cmap=cmap_t, vmin=-5, vmax=30)
    x1, x2, y1, y2 = 250, 290, 0, 40
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    rectpatch, connects=ax30.indicate_inset_zoom(axins, edgecolor="k", alpha=1, linewidth=0.5)
    connects[0].set(linewidth=0.5)
    connects[1].set(linewidth=0.5)
    connects[2].set(linewidth=0.5, visible=True)
    connects[3].set(linewidth=0.5, visible=True)

    ax29.set_xticks([45, 135, 225, 315])
    ax29.set_yticks([-90, 0, 90])
    ax30.set_xticks([45, 135, 225, 315])
    ax30.set_yticks([])
    ax31.set_xticks([45, 135, 225, 315])
    ax31.set_yticks([])
    ax32.set_xticks([45, 135, 225, 315])
    ax32.set_yticks([])

    ax29.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax30.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax31.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax32.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    
    ax29.set_ylabel("Latitude $[^{\\circ}]$", labelpad=-3)
    ax29.set_xlabel("Longitude $[^{\\circ}]$", labelpad=1)
    ax30.set_xlabel("Longitude $[^{\\circ}]$", labelpad=1)
    ax31.set_xlabel("Longitude $[^{\\circ}]$", labelpad=1)
    ax32.set_xlabel("Longitude $[^{\\circ}]$", labelpad=1)

    ax33 = figure.add_axes([0.95, 0.25, 0.01, 0.40])
    ax33.tick_params(axis="both", direction="out", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=2)
  
    cbar33 = figure.colorbar(im[3], cax=ax33, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[0, 15, 30])
    cbar33.ax.set_title("$T$ $[^{\\circ}C]$")

    figure.savefig(f"../figs/nature04.pdf", dpi=1000)

    print('Hello')

    return


if __name__ == '__main__':

    main()