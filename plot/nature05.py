# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 16:42:24 2021
@author: guemesturb
"""


import os
import re
import cmasher
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
    fig_height = fig_width * 0.45 

    figure = plt.figure('panel01', figsize=(fig_width, fig_height))
    im = []

    ax04 = figure.add_axes([0.01, 0.850, 0.1, 0.05])
    ax04.text(0.2, 0.5, 'Test Case 4 -- Experimental Turbulent Boundary Layer at $Re_{\\tau}\\approx1000$', fontsize=8)
    ax04.axis("off")

    ax05 = figure.add_axes([0.040, 0.95, 0.21, 0.04])
    ax06 = figure.add_axes([0.260, 0.95, 0.21, 0.04])
    ax07 = figure.add_axes([0.480, 0.95, 0.21, 0.04])
    ax08 = figure.add_axes([0.700, 0.95, 0.21, 0.04])
    ax05.text(0.5, 0.5, 'LR Input',              horizontalalignment='center', verticalalignment='center', transform=ax05.transAxes, fontsize=8)
    ax06.text(0.5, 0.5, 'Sparse HR Reference',   horizontalalignment='center', verticalalignment='center', transform=ax06.transAxes, fontsize=8)
    ax07.text(0.5, 0.5, 'RaSeedGAN',    horizontalalignment='center', verticalalignment='center', transform=ax07.transAxes, fontsize=8)
    ax08.text(0.5, 0.5, 'Complete HR Reference', horizontalalignment='center', verticalalignment='center', transform=ax08.transAxes, fontsize=8)
    ax05.axis("off")
    ax06.axis("off")
    ax07.axis("off")
    ax08.axis("off")

    """
        TBL
    """

    cmap_u = plt.get_cmap('cmr.iceburn').copy()
    cmap_v = plt.get_cmap('cmr.redshift').copy()
    # cmap_v = matplotlib.cm.get_cmap("RdBu_r").copy()
    cmap_u.set_bad(color='w')
    cmap_v.set_bad(color='w')

    ax39 = figure.add_axes([0.040, 0.470, 0.210, 0.370])
    ax40 = figure.add_axes([0.260, 0.470, 0.210, 0.370])
    ax41 = figure.add_axes([0.480, 0.470, 0.210, 0.370])
    ax42 = figure.add_axes([0.700, 0.470, 0.210, 0.370])
    ax43 = figure.add_axes([0.040, 0.070, 0.210, 0.370])
    ax44 = figure.add_axes([0.260, 0.070, 0.210, 0.370])
    ax45 = figure.add_axes([0.480, 0.070, 0.210, 0.370])
    ax46 = figure.add_axes([0.700, 0.070, 0.210, 0.370])

    filename = f"/storage/aguemes/gan-piv/exptbl/ss04/results/predictions_architecture-01-noise-000.npz"
    data = np.load(filename)
    
    hr_predic  = data['hr_predic'] / 48440 / 0.000017 / 15.5576
    hr_target  = data['hr_target'] / 48440 / 0.000017 / 15.5576
    lr_target  = data['lr_target'] /  48440 / 0.000017 / 15.5576
    dns_target  = data['dns_target'] / 48440 / 0.000017 / 15.5576
    fl_target  = data['fl_target']
    hr_target[np.repeat(fl_target, 2, axis=3) == 0] = np.nan
    xhr = data['xhr'] / 48440 / 0.0255
    yhr = data['yhr'] / 48440 / 0.0255

    Qdns = dns_target - np.mean(dns_target, axis=(0,))
    Qpiv = lr_target - np.mean(lr_target, axis=(0,))
    Qgan = hr_predic - np.mean(hr_predic, axis=(0,))
    Qdns = -Qdns[:, :, :, 0] * Qdns[:, :, :, 1]
    Qpiv = -Qpiv[:, :, :, 0] * Qpiv[:, :, :, 1]
    Qgan = -Qgan[:, :, :, 0] * Qgan[:, :, :, 1]
    th_dns = 1.75 * np.std(dns_target[:, :, :, 0], axis=(0,)) * np.std(dns_target[:, :, :, 1], axis=(0,))
    th_piv = 1.75 * np.std(lr_target[:, :, :, 0], axis=(0,)) * np.std(lr_target[:, :, :, 1], axis=(0,))
    th_gan = 1.75 * np.std(hr_predic[:, :, :, 0], axis=(0,)) * np.std(hr_predic[:, :, :, 1], axis=(0,))
    Qdns = np.where(Qdns > th_dns, 1, 0)
    Qgan = np.where(Qgan > th_gan, 1, 0)
    Qpiv = np.where(Qpiv > th_piv, 1, 0)

    dns_target = dns_target - np.mean(dns_target, axis=(0,))
    lr_target = lr_target - np.mean(lr_target, axis=(0,))
    hr_target = hr_target - np.nanmean(hr_target, axis=(0,))
    hr_predic = hr_predic - np.mean(hr_predic, axis=(0,))

    print(np.argmax(np.sum(Qdns, axis=(1,2))))

    idx = 538
    idx = 510
    
    im.append(ax39.imshow(lr_target[idx,:,:,0], origin="upper", extent=[0, xhr.max(), 0, yhr.max()], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax39.contour(Qpiv[idx,:,:], levels=[0.99], origin="upper", colors='w', linewidths=1, alpha=1.0, extent=[0, xhr.max(), 0, yhr.max()]))
    im.append(ax40.imshow(hr_target[idx,:,:,0], origin="upper", extent=[0, xhr.max(), 0, yhr.max()], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax41.imshow(hr_predic[idx,:,:,0], origin="upper", extent=[0, xhr.max(), 0, yhr.max()], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax41.contour(Qgan[idx,:,:], levels=[0.99], origin="upper", colors='w', linewidths=1, alpha=1.0, extent=[0, xhr.max(), 0, yhr.max()]))
    im.append(ax42.imshow(dns_target[idx,:,:,0], origin="upper", extent=[0, xhr.max(), 0, yhr.max()], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax42.contour(Qdns[idx,:,:], levels=[0.99], origin="upper", colors='w', linewidths=1, alpha=1.0, extent=[0, xhr.max(), 0, yhr.max()]))

    im.append(ax43.imshow(lr_target[idx,:,:,1], origin="upper", extent=[0, xhr.max(), 0, yhr.max()], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax43.contour(Qpiv[idx,:,:], levels=[0.99], origin="upper", colors='w', linewidths=1, alpha=1.0, extent=[0, xhr.max(), 0, yhr.max()]))
    im.append(ax44.imshow(hr_target[idx,:,:,1], origin="upper", extent=[0, xhr.max(), 0, yhr.max()], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax45.imshow(hr_predic[idx,:,:,1], origin="upper", extent=[0, xhr.max(), 0, yhr.max()], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax45.contour(Qgan[idx,:,:], levels=[0.99], origin="upper", colors='w', linewidths=1, alpha=1.0, extent=[0, xhr.max(), 0, yhr.max()]))
    im.append(ax46.imshow(dns_target[idx,:,:,1], origin="upper", extent=[0, xhr.max(), 0, yhr.max()], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax46.contour(Qdns[idx,:,:], levels=[0.99], origin="upper", colors='w', linewidths=1, alpha=1.0, extent=[0, xhr.max(), 0, yhr.max()]))

    ax39.set_xticks([])
    ax39.set_yticks([0, 0.5, 1, 1.5])
    ax40.set_xticks([])
    ax40.set_yticks([])
    ax41.set_xticks([])
    ax41.set_yticks([])
    ax42.set_xticks([])
    ax42.set_yticks([])

    ax43.set_xticks([0, 0.5, 1, 1.5])
    ax43.set_yticks([0, 0.5, 1, 1.5])
    ax44.set_xticks([0, 0.5, 1, 1.5])
    ax44.set_yticks([])
    ax45.set_xticks([0, 0.5, 1, 1.5])
    ax45.set_yticks([])
    ax46.set_xticks([0, 0.5, 1, 1.5])
    ax46.set_yticks([])

    ax39.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax40.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax41.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax42.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax43.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax44.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax45.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax46.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)


    ax47 = figure.add_axes([0.93, 0.490, 0.01, 0.3])
    ax48 = figure.add_axes([0.93, 0.090, 0.01, 0.3])
    ax47.tick_params(axis="both", direction="out", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=2)
    ax48.tick_params(axis="both", direction="out", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=2)
    
    cbar47 = figure.colorbar(im[3], cax=ax47, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-0.2, 0.0, 0.2])
    cbar48 = figure.colorbar(im[7], cax=ax48, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-0.1, 0, 0.1])
    cbar47.ax.set_title("$u/U_{\\infty}$")
    cbar48.ax.set_title("$v/U_{\\infty}$")
    [(t.set_horizontalalignment('right'), t.set_x(4)) for t in cbar47.ax.get_yticklabels()]
    [(t.set_horizontalalignment('right'), t.set_x(4)) for t in cbar48.ax.get_yticklabels()]
    # # [(t.set_horizontalalignment('right'), t.set_x(3)) for t in cbar38.ax.get_yticklabels()]
    # [(t.set_horizontalalignment('right'), t.set_x(4)) for t in cbar17.ax.get_yticklabels()]
    # [(t.set_horizontalalignment('right'), t.set_x(4)) for t in cbar18.ax.get_yticklabels()]
    # [(t.set_horizontalalignment('right'), t.set_x(4)) for t in cbar27.ax.get_yticklabels()]
    # [(t.set_horizontalalignment('right'), t.set_x(4)) for t in cbar28.ax.get_yticklabels()]
    # [(t.set_horizontalalignment('right'), t.set_x(3)) for t in cbar33.ax.get_yticklabels()]
    

    ax39.set_ylabel("$y/\\delta_{99}$", labelpad=2)
    ax43.set_xlabel("$x/\\delta_{99}$", labelpad=0)
    ax43.set_ylabel("$y/\\delta_{99}$", labelpad=2)
    ax44.set_xlabel("$x/\\delta_{99}$", labelpad=0)
    ax45.set_xlabel("$x/\\delta_{99}$", labelpad=0)
    ax46.set_xlabel("$x/\\delta_{99}$", labelpad=0)

    figure.savefig(f"../figs/nature05.pdf", dpi=1000)

    print('Hello')

    return


if __name__ == '__main__':

    main()