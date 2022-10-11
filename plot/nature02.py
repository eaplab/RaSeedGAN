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
    fig_height = fig_width * 0.45 

    figure = plt.figure('panel01', figsize=(fig_width, fig_height))
    im = []

    ax01 = figure.add_axes([0.01, 0.900, 0.1, 0.05])
    ax02 = figure.add_axes([0.01, 0.450, 0.1, 0.05])
    ax01.text(0.2, 0.5, 'Test Case 1 -- Fluidic Pinball at $Re_D=130$', fontsize=8)
    ax02.text(0.2, 0.5, 'Test Case 2 -- Turbulent Channel Flow at $Re_{\\tau}=1000$', fontsize=8)
    ax01.axis("off")
    ax02.axis("off")

    ax05 = figure.add_axes([0.040, 0.96, 0.21, 0.04])
    ax06 = figure.add_axes([0.260, 0.96, 0.21, 0.04])
    ax07 = figure.add_axes([0.480, 0.96, 0.21, 0.04])
    ax08 = figure.add_axes([0.700, 0.96, 0.21, 0.04])
    ax05.text(0.5, 0.5, 'LR Input',              horizontalalignment='center', verticalalignment='center', transform=ax05.transAxes, fontsize=8)
    ax06.text(0.5, 0.5, 'Sparse HR Reference',   horizontalalignment='center', verticalalignment='center', transform=ax06.transAxes, fontsize=8)
    ax07.text(0.5, 0.5, 'RaSeedGAN',    horizontalalignment='center', verticalalignment='center', transform=ax07.transAxes, fontsize=8)
    ax08.text(0.5, 0.5, 'Complete HR Reference', horizontalalignment='center', verticalalignment='center', transform=ax08.transAxes, fontsize=8)
    ax05.axis("off")
    ax06.axis("off")
    ax07.axis("off")
    ax08.axis("off")

    """
        Pinball
    """

    ax09 = figure.add_axes([0.040, 0.750, 0.210, 0.15])
    ax10 = figure.add_axes([0.260, 0.750, 0.210, 0.15])
    ax11 = figure.add_axes([0.480, 0.750, 0.210, 0.15])
    ax12 = figure.add_axes([0.700, 0.750, 0.210, 0.15])
    ax13 = figure.add_axes([0.040, 0.570, 0.210, 0.15])
    ax14 = figure.add_axes([0.260, 0.570, 0.210, 0.15])
    ax15 = figure.add_axes([0.480, 0.570, 0.210, 0.15])
    ax16 = figure.add_axes([0.700, 0.570, 0.210, 0.15])

    filename = f"/storage/aguemes/gan-piv/pinball/ss08/results/predictions_architecture-01-noise-010.npz"
    data = np.load(filename)
    dns_target = data['dns_target'] 
    hr_predic  = data['hr_predic'] 
    hr_target  = data['hr_target'] 
    lr_target  = data['lr_target'] 
    fl_target  = data['fl_target']
    hr_target[np.repeat(fl_target, 2, axis=3) == 0] = np.nan
    xlr = data['xlr'] 
    xhr = data['xhr'] 
    yhr = data['yhr'] 

    Res = 25
    xmin = -5
    ymin = -4 + 4 / Res
    xhr = xhr / Res + xmin
    xlr = xlr / Res + xmin
    yhr = yhr / Res + ymin

    # cmap_u = matplotlib.cm.get_cmap("RdYlGn").copy()
    cmap_u = plt.get_cmap('cmr.iceburn').copy()
    cmap_v = plt.get_cmap('cmr.redshift').copy()
    # cmap_v = matplotlib.cm.get_cmap("RdBu_r").copy()
    cmap_u.set_bad(color='w')
    cmap_v.set_bad(color='w')

    im.append(ax09.imshow(lr_target[0,:,:,0], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_u, vmin=-1.5, vmax=1.5))
    im.append(ax10.imshow(hr_target[0,:,:,0], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_u, vmin=-1.5, vmax=1.5))
    im.append(ax11.imshow(hr_predic[0,:,:,0], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_u, vmin=-1.5, vmax=1.5))
    im.append(ax12.imshow(dns_target[0,:,:,0], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_u, vmin=-1.5, vmax=1.5))

    im.append(ax13.imshow(lr_target[0,:,:,1], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_v, vmin=-0.7, vmax=0.7))
    im.append(ax14.imshow(hr_target[0,:,:,1], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_v, vmin=-0.7, vmax=0.7))
    im.append(ax15.imshow(hr_predic[0,:,:,1], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_v, vmin=-0.7, vmax=0.7))
    im.append(ax16.imshow(dns_target[0,:,:,1], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_v, vmin=-0.7, vmax=0.7))

    axins = ax10.inset_axes([0.4, 0.1, 0.8, 0.8])
    axins.imshow(hr_target[0,:,:,0], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_u, vmin=-1.5, vmax=1.5)
    draw_circle1 = plt.Circle((- 1.299, 0), 0.5, edgecolor=None, facecolor='gray')
    draw_circle2 = plt.Circle((0, 0.75), 0.5,    edgecolor=None, facecolor='gray')
    draw_circle3 = plt.Circle((0, -0.75), 0.5,   edgecolor=None, facecolor='gray')
    axins.add_artist(draw_circle1)
    axins.add_artist(draw_circle2)
    axins.add_artist(draw_circle3)
    x1, x2, y1, y2 = 0, 3, -1.5, 1.5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    rectpatch, connects=ax10.indicate_inset_zoom(axins, edgecolor="k", alpha=1, linewidth=0.5)
    connects[0].set(linewidth=0.5)
    connects[1].set(linewidth=0.5)
    connects[2].set(linewidth=0.5)
    connects[3].set(linewidth=0.5)

    axins = ax14.inset_axes([0.4, 0.1, 0.8, 0.8])
    axins.imshow(hr_target[0,:,:,1], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_v, vmin=-0.7, vmax=0.7)
    draw_circle1 = plt.Circle((- 1.299, 0), 0.5, edgecolor=None, facecolor='gray')
    draw_circle2 = plt.Circle((0, 0.75), 0.5,    edgecolor=None, facecolor='gray')
    draw_circle3 = plt.Circle((0, -0.75), 0.5,   edgecolor=None, facecolor='gray')
    axins.add_artist(draw_circle1)
    axins.add_artist(draw_circle2)
    axins.add_artist(draw_circle3)
    x1, x2, y1, y2 = 0, 3, -1.5, 1.5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    rectpatch, connects=ax14.indicate_inset_zoom(axins, edgecolor="k", alpha=1, linewidth=0.5)
    connects[0].set(linewidth=0.5)
    connects[1].set(linewidth=0.5)
    connects[2].set(linewidth=0.5)
    connects[3].set(linewidth=0.5)

    draw_circle1 = plt.Circle((- 1.299, 0), 0.5, edgecolor=None, facecolor='gray')
    draw_circle2 = plt.Circle((0, 0.75), 0.5,    edgecolor=None, facecolor='gray')
    draw_circle3 = plt.Circle((0, -0.75), 0.5,   edgecolor=None, facecolor='gray')
    ax09.add_artist(draw_circle1)
    ax09.add_artist(draw_circle2)
    ax09.add_artist(draw_circle3)
    draw_circle1 = plt.Circle((- 1.299, 0), 0.5, edgecolor=None, facecolor='gray')
    draw_circle2 = plt.Circle((0, 0.75), 0.5,    edgecolor=None, facecolor='gray')
    draw_circle3 = plt.Circle((0, -0.75), 0.5,   edgecolor=None, facecolor='gray')
    ax10.add_artist(draw_circle1)
    ax10.add_artist(draw_circle2)
    ax10.add_artist(draw_circle3)
    draw_circle1 = plt.Circle((- 1.299, 0), 0.5, edgecolor=None, facecolor='gray')
    draw_circle2 = plt.Circle((0, 0.75), 0.5,    edgecolor=None, facecolor='gray')
    draw_circle3 = plt.Circle((0, -0.75), 0.5,   edgecolor=None, facecolor='gray')
    ax11.add_artist(draw_circle1)
    ax11.add_artist(draw_circle2)
    ax11.add_artist(draw_circle3)
    draw_circle1 = plt.Circle((- 1.299, 0), 0.5, edgecolor=None, facecolor='gray')
    draw_circle2 = plt.Circle((0, 0.75), 0.5,    edgecolor=None, facecolor='gray')
    draw_circle3 = plt.Circle((0, -0.75), 0.5,   edgecolor=None, facecolor='gray')
    ax12.add_artist(draw_circle1)
    ax12.add_artist(draw_circle2)
    ax12.add_artist(draw_circle3)
    draw_circle1 = plt.Circle((- 1.299, 0), 0.5, edgecolor=None, facecolor='gray')
    draw_circle2 = plt.Circle((0, 0.75), 0.5,    edgecolor=None, facecolor='gray')
    draw_circle3 = plt.Circle((0, -0.75), 0.5,   edgecolor=None, facecolor='gray')
    ax13.add_artist(draw_circle1)
    ax13.add_artist(draw_circle2)
    ax13.add_artist(draw_circle3)
    draw_circle1 = plt.Circle((- 1.299, 0), 0.5, edgecolor=None, facecolor='gray')
    draw_circle2 = plt.Circle((0, 0.75), 0.5,    edgecolor=None, facecolor='gray')
    draw_circle3 = plt.Circle((0, -0.75), 0.5,   edgecolor=None, facecolor='gray')
    ax14.add_artist(draw_circle1)
    ax14.add_artist(draw_circle2)
    ax14.add_artist(draw_circle3)
    draw_circle1 = plt.Circle((- 1.299, 0), 0.5, edgecolor=None, facecolor='gray')
    draw_circle2 = plt.Circle((0, 0.75), 0.5,    edgecolor=None, facecolor='gray')
    draw_circle3 = plt.Circle((0, -0.75), 0.5,   edgecolor=None, facecolor='gray')
    ax15.add_artist(draw_circle1)
    ax15.add_artist(draw_circle2)
    ax15.add_artist(draw_circle3)
    draw_circle1 = plt.Circle((- 1.299, 0), 0.5, edgecolor=None, facecolor='gray')
    draw_circle2 = plt.Circle((0, 0.75), 0.5,    edgecolor=None, facecolor='gray')
    draw_circle3 = plt.Circle((0, -0.75), 0.5,   edgecolor=None, facecolor='gray')
    ax16.add_artist(draw_circle1)
    ax16.add_artist(draw_circle2)
    ax16.add_artist(draw_circle3)

    ax09.set_xticks([])
    ax09.set_yticks([-3, 0, 3])
    ax10.set_xticks([])
    ax10.set_yticks([])
    ax11.set_xticks([])
    ax11.set_yticks([])
    ax12.set_xticks([])
    ax12.set_yticks([])

    ax13.set_xticks([-5, 0, 5, 10, 15])
    ax13.set_yticks([-3, 0, 3])
    ax14.set_xticks([-5, 0, 5, 10, 15])
    ax14.set_yticks([])
    ax15.set_xticks([-5, 0, 5, 10, 15])
    ax15.set_yticks([])
    ax16.set_xticks([-5, 0, 5, 10, 15])
    ax16.set_yticks([])

    ax09.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax10.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax11.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax12.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax13.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax14.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax15.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax16.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)


    ax17 = figure.add_axes([0.94, 0.750, 0.01, 0.11])
    ax18 = figure.add_axes([0.94, 0.570, 0.01, 0.11])
    ax17.tick_params(axis="both", direction="out", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=2)
    ax18.tick_params(axis="both", direction="out", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=2)

    cbar17 = figure.colorbar(im[0], cax=ax17, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-1.5, 0.0, 1.5])
    cbar18 = figure.colorbar(im[4], cax=ax18, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-0.7, 0, 0.7])
    cbar17.ax.set_title("$u/U_{\\infty}$")
    cbar18.ax.set_title("$v/U_{\\infty}$")

    ax09.set_ylabel("$y/D$", labelpad=0)
    ax13.set_xlabel("$x/D$", labelpad=0)
    ax13.set_ylabel("$y/D$", labelpad=0)
    ax14.set_xlabel("$x/D$", labelpad=0)
    ax15.set_xlabel("$x/D$", labelpad=0)
    ax16.set_xlabel("$x/D$", labelpad=0)

    """
        Channel
    """

    ax19 = figure.add_axes([0.040, 0.270, 0.210, 0.180])
    ax20 = figure.add_axes([0.260, 0.270, 0.210, 0.180])
    ax21 = figure.add_axes([0.480, 0.270, 0.210, 0.180])
    ax22 = figure.add_axes([0.700, 0.270, 0.210, 0.180])
    ax23 = figure.add_axes([0.040, 0.060, 0.210, 0.180])
    ax24 = figure.add_axes([0.260, 0.060, 0.210, 0.180])
    ax25 = figure.add_axes([0.480, 0.060, 0.210, 0.180])
    ax26 = figure.add_axes([0.700, 0.060, 0.210, 0.180])

    filename = f"/storage/aguemes/gan-piv/channel/ss08/results/predictions_architecture-01-noise-010.npz"
    data = np.load(filename)
    dns_target = data['dns_target'] * 1/512/0.013
    hr_predic  = data['hr_predic'] 
    hr_target  = data['hr_target'] 
    lr_target  = data['lr_target'] 
    fl_target  = data['fl_target']
    hr_target[np.repeat(fl_target, 2, axis=3) == 0] = np.nan
    # cmap_u = matplotlib.cm.get_cmap("RdGy").copy()
    # cmap_u.set_bad(color='w')

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

    im.append(ax19.imshow(lr_target[985,:,:,0], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax19.contour(Qpiv[985,:,:], levels=[0.99], origin="upper", colors='w', linewidths=1, alpha=1.0, extent=[0, 2, 0, 1]))
    im.append(ax20.imshow(hr_target[985,:,:,0], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax21.imshow(hr_predic[985,:,:,0], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax21.contour(Qgan[985,:,:], levels=[0.99], origin="upper", colors='w', linewidths=1, alpha=1.0, extent=[0, 2, 0, 1]))
    im.append(ax22.imshow(dns_target[985,:,:,0], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax22.contour(Qdns[985,:,:], levels=[0.99], origin="upper", colors='w', linewidths=1, alpha=1.0, extent=[0, 2, 0, 1]))

    im.append(ax23.imshow(lr_target[985,:,:,1], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax23.contour(Qpiv[985,:,:], levels=[0.99], origin="upper", colors='w', linewidths=1, alpha=1.0, extent=[0, 2, 0, 1]))
    im.append(ax24.imshow(hr_target[985,:,:,1], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax25.imshow(hr_predic[985,:,:,1], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax25.contour(Qgan[985,:,:], levels=[0.99], origin="upper", colors='w', linewidths=1, alpha=1.0, extent=[0, 2, 0, 1]))
    im.append(ax26.imshow(dns_target[985,:,:,1], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax26.contour(Qdns[985,:,:], levels=[0.99], origin="upper", colors='w', linewidths=1, alpha=1.0, extent=[0, 2, 0, 1]))

    axins = ax20.inset_axes([0.25, 0.1, 0.7, 0.7])
    axins.imshow(hr_target[985,:,:,0], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_u, vmin=-0.2, vmax=0.2)
    x1, x2, y1, y2 = 0.1, 0.4, 0, 0.15
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    rectpatch, connects=ax20.indicate_inset_zoom(axins, edgecolor="k", alpha=1, linewidth=0.5)
    connects[0].set(linewidth=0.5)
    connects[1].set(linewidth=0.5)
    connects[2].set(linewidth=0.5)
    connects[3].set(linewidth=0.5)

    axins = ax24.inset_axes([0.25, 0.1, 0.7, 0.7])
    axins.imshow(hr_target[985,:,:,1], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_v, vmin=-0.1, vmax=0.1)
    x1, x2, y1, y2 = 0.1, 0.4, 0, 0.15
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    rectpatch, connects=ax24.indicate_inset_zoom(axins, edgecolor="k", alpha=1, linewidth=0.5)
    connects[0].set(linewidth=0.5)
    connects[1].set(linewidth=0.5)
    connects[2].set(linewidth=0.5)
    connects[3].set(linewidth=0.5)

    ax19.set_xticks([])
    ax19.set_yticks([0, 0.5, 1])
    ax20.set_yticks([])
    ax20.set_xticks([])
    ax21.set_xticks([])
    ax21.set_yticks([])
    ax22.set_xticks([])
    ax22.set_yticks([])

    ax23.set_xticks([0, 1, 2])
    ax23.set_yticks([0, 0.5, 1])
    ax24.set_xticks([0, 1, 2])
    ax24.set_yticks([])
    ax25.set_xticks([0, 1, 2])
    ax25.set_yticks([])
    ax26.set_xticks([0, 1, 2])
    ax26.set_yticks([])

    ax19.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax20.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax21.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax22.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax23.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax24.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax25.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax26.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)

    ax19.set_ylabel("$y/h$", labelpad=0.5)
    ax23.set_xlabel("$x/h$", labelpad=0)
    ax23.set_ylabel("$y/h$", labelpad=0.5)
    ax24.set_xlabel("$x/h$", labelpad=0)
    ax25.set_xlabel("$x/h$", labelpad=0)
    ax26.set_xlabel("$x/h$", labelpad=0)

    ax27 = figure.add_axes([0.94, 0.290, 0.01, 0.11])
    ax28 = figure.add_axes([0.94, 0.080, 0.01, 0.11])
    ax27.tick_params(axis="both", direction="out", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=2)
    ax28.tick_params(axis="both", direction="out", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=2)

    cbar27 = figure.colorbar(im[8], cax=ax27, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-0.2, 0.0, 0.2])
    cbar28 = figure.colorbar(im[15], cax=ax28, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-0.1, 0, 0.1])
    cbar27.ax.set_title("$u/U_b$")
    cbar28.ax.set_title("$v/U_b$")

    figure.savefig(f"../figs/nature02.pdf", dpi=1000)

    print('Hello')

    return


if __name__ == '__main__':

    main()