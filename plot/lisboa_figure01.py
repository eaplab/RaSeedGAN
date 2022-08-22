# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 16:42:24 2021
@author: guemesturb
"""


import os
import re
import sys
sys.path.insert(1, '/home/aguemes/tools/TheArtist')
import matplotlib
import numpy as np
from artist import TheArtist
import matplotlib.pyplot as plt


def main():

    plt.rc('text', usetex='True')
    plt.rc('font', family='Serif', size='6')
    # plt.rc('axes', titlesize='6')

    fig_width_pt = 510
    inches_per_pt = 1.0 / 72.27  
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width * 1.0 

    figure = plt.figure('panel01', figsize=(fig_width, fig_height))
    im = []

    ax01 = figure.add_axes([0.01, 0.920, 0.1, 0.05])
    ax02 = figure.add_axes([0.01, 0.710, 0.1, 0.05])
    ax03 = figure.add_axes([0.01, 0.435, 0.1, 0.05])
    ax04 = figure.add_axes([0.01, 0.275, 0.1, 0.05])
    ax01.text(0.2, 0.5, 'Test Case 1 -- Fluidic Pinball at $Re_D=100$', fontsize=8)
    ax02.text(0.2, 0.5, 'Test Case 2 -- Turbulent Channel Flow at $Re_{\\tau}=1000$', fontsize=8)
    ax03.text(0.2, 0.5, 'Test Case 3 -- NOAA Sea Surface Temperature', fontsize=8)
    ax04.text(0.2, 0.5, 'Test Case 4 -- Experimental Turbulent Boundary Layer at $Re_{\\tau}\\approx1000$', fontsize=8)
    ax01.axis("off")
    ax02.axis("off")
    ax03.axis("off")
    ax04.axis("off")

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

    ax09 = figure.add_axes([0.040, 0.860, 0.210, 0.075])
    ax10 = figure.add_axes([0.260, 0.860, 0.210, 0.075])
    ax11 = figure.add_axes([0.480, 0.860, 0.210, 0.075])
    ax12 = figure.add_axes([0.700, 0.860, 0.210, 0.075])
    ax13 = figure.add_axes([0.040, 0.780, 0.210, 0.075])
    ax14 = figure.add_axes([0.260, 0.780, 0.210, 0.075])
    ax15 = figure.add_axes([0.480, 0.780, 0.210, 0.075])
    ax16 = figure.add_axes([0.700, 0.780, 0.210, 0.075])

    filename = f"/STORAGE01/aguemes/gan-piv/pinball/ss08/results/predictions_architecture-01-noise-010.npz"
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

    cmap_u = matplotlib.cm.get_cmap("seismic").copy()
    cmap_v = matplotlib.cm.get_cmap("PiYG").copy()
    cmap_u.set_bad(color='k')
    cmap_v.set_bad(color='k')

    im.append(ax09.imshow(lr_target[0,:,:,0], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_u, vmin=-1.5, vmax=1.5))
    im.append(ax10.imshow(hr_target[0,:,:,0], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_u, vmin=-1.5, vmax=1.5))
    im.append(ax11.imshow(hr_predic[0,:,:,0], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_u, vmin=-1.5, vmax=1.5))
    im.append(ax12.imshow(dns_target[0,:,:,0], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_u, vmin=-1.5, vmax=1.5))

    im.append(ax13.imshow(lr_target[0,:,:,1], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_v, vmin=-0.7, vmax=0.7))
    im.append(ax14.imshow(hr_target[0,:,:,1], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_v, vmin=-0.7, vmax=0.7))
    im.append(ax15.imshow(hr_predic[0,:,:,1], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_v, vmin=-0.7, vmax=0.7))
    im.append(ax16.imshow(dns_target[0,:,:,1], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_v, vmin=-0.7, vmax=0.7))

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


    ax17 = figure.add_axes([0.94, 0.870, 0.01, 0.05])
    ax18 = figure.add_axes([0.94, 0.780, 0.01, 0.05])
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

    ax19 = figure.add_axes([0.040, 0.620, 0.210, 0.10])
    ax20 = figure.add_axes([0.260, 0.620, 0.210, 0.10])
    ax21 = figure.add_axes([0.480, 0.620, 0.210, 0.10])
    ax22 = figure.add_axes([0.700, 0.620, 0.210, 0.10])
    ax23 = figure.add_axes([0.040, 0.505, 0.210, 0.10])
    ax24 = figure.add_axes([0.260, 0.505, 0.210, 0.10])
    ax25 = figure.add_axes([0.480, 0.505, 0.210, 0.10])
    ax26 = figure.add_axes([0.700, 0.505, 0.210, 0.10])

    filename = f"/STORAGE01/aguemes/gan-piv/channel/ss08/results/predictions_architecture-01-noise-010.npz"
    data = np.load(filename)
    dns_target = data['dns_target'] * 1/512/0.013
    hr_predic  = data['hr_predic'] 
    hr_target  = data['hr_target'] 
    lr_target  = data['lr_target'] 
    fl_target  = data['fl_target']
    hr_target[np.repeat(fl_target, 2, axis=3) == 0] = np.nan
    cmap_u = matplotlib.cm.get_cmap("seismic").copy()
    cmap_u.set_bad(color='k')

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
    im.append(ax19.contour(Qpiv[985,:,:], levels=[0.99], origin="upper", colors='k', linewidths=1, alpha=0.5, extent=[0, 2, 0, 1]))
    im.append(ax20.imshow(hr_target[985,:,:,0], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax21.imshow(hr_predic[985,:,:,0], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax21.contour(Qgan[985,:,:], levels=[0.99], origin="upper", colors='k', linewidths=1, alpha=0.5, extent=[0, 2, 0, 1]))
    im.append(ax22.imshow(dns_target[985,:,:,0], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax22.contour(Qdns[985,:,:], levels=[0.99], origin="upper", colors='k', linewidths=1, alpha=0.5, extent=[0, 2, 0, 1]))

    im.append(ax23.imshow(lr_target[985,:,:,1], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax23.contour(Qpiv[985,:,:], levels=[0.99], origin="upper", colors='k', linewidths=1, alpha=0.5, extent=[0, 2, 0, 1]))
    im.append(ax24.imshow(hr_target[985,:,:,1], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax25.imshow(hr_predic[985,:,:,1], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax25.contour(Qgan[985,:,:], levels=[0.99], origin="upper", colors='k', linewidths=1, alpha=0.5, extent=[0, 2, 0, 1]))
    im.append(ax26.imshow(dns_target[985,:,:,1], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax26.contour(Qdns[985,:,:], levels=[0.99], origin="upper", colors='k', linewidths=1, alpha=0.5, extent=[0, 2, 0, 1]))

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

    ax27 = figure.add_axes([0.94, 0.625, 0.01, 0.08])
    ax28 = figure.add_axes([0.94, 0.510, 0.01, 0.08])
    ax27.tick_params(axis="both", direction="out", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=2)
    ax28.tick_params(axis="both", direction="out", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=2)

    cbar27 = figure.colorbar(im[8], cax=ax27, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-0.2, 0.0, 0.2])
    cbar28 = figure.colorbar(im[15], cax=ax28, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-0.1, 0, 0.1])
    cbar27.ax.set_title("$u/U_b$")
    cbar28.ax.set_title("$v/U_b$")

    """
        SST
    """

    ax29 = figure.add_axes([0.040, 0.35, 0.210, 0.10])
    ax30 = figure.add_axes([0.260, 0.35, 0.210, 0.10])
    ax31 = figure.add_axes([0.480, 0.35, 0.210, 0.10])
    ax32 = figure.add_axes([0.700, 0.35, 0.210, 0.10])

    filename = f"/STORAGE01/aguemes/gan-piv/sst/ss08/results/predictions_architecture-01-noise-000.npz"
    data = np.load(filename)
    dns_target = data['dns_target'] 
    hr_predic  = data['hr_predic'] 
    hr_target  = data['hr_target'] 
    lr_target  = data['lr_target'] 
    fl_target  = data['fl_target']
    hr_target[fl_target == 0] = np.nan

    cmap_t = matplotlib.cm.get_cmap("viridis").copy()
    cmap_t.set_bad(color='k')

    im.append(ax29.imshow(lr_target[0,:,:,0], origin="lower", extent=[0, 360, -90, 90], cmap=cmap_t, vmin=-5, vmax=30))
    im.append(ax30.imshow(hr_target[0,:,:,0], origin="lower", extent=[0, 360, -90, 90], cmap=cmap_t, vmin=-5, vmax=30))
    im.append(ax31.imshow(hr_predic[0,:,:,0], origin="lower", extent=[0, 360, -90, 90], cmap=cmap_t, vmin=-5, vmax=30))
    im.append(ax32.imshow(dns_target[0,:,:,0], origin="lower", extent=[0, 360, -90, 90], cmap=cmap_t, vmin=-5, vmax=30))

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

    ax33 = figure.add_axes([0.95, 0.355, 0.01, 0.08])
    ax33.tick_params(axis="both", direction="out", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=2)
  
    cbar33 = figure.colorbar(im[22], cax=ax33, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[0, 15, 30])
    cbar33.ax.set_title("$T$ $[^{\\circ}C]$")

    """
        TBL
    """

    ax39 = figure.add_axes([0.040, 0.165, 0.210, 0.125])
    ax40 = figure.add_axes([0.260, 0.165, 0.210, 0.125])
    ax41 = figure.add_axes([0.480, 0.165, 0.210, 0.125])
    ax42 = figure.add_axes([0.700, 0.165, 0.210, 0.125])
    ax43 = figure.add_axes([0.040, 0.030, 0.210, 0.125])
    ax44 = figure.add_axes([0.260, 0.030, 0.210, 0.125])
    ax45 = figure.add_axes([0.480, 0.030, 0.210, 0.125])
    ax46 = figure.add_axes([0.700, 0.030, 0.210, 0.125])

    filename = f"/STORAGE01/aguemes/gan-piv/exptbl/ss04/results/predictions_architecture-01-noise-000.npz"
    data = np.load(filename)
    
    hr_predic  = data['hr_predic'] / 48440 / 0.000017 / 15.5576
    hr_target  = data['hr_target'] / 48440 / 0.000017 / 15.5576
    lr_target  = data['lr_target'] /  48440 / 0.000017 / 15.5576
    dns_target  = data['dns_target'] / 48440 / 0.000017 / 15.5576
    fl_target  = data['fl_target']
    hr_target[np.repeat(fl_target, 2, axis=3) == 0] = np.nan
    xhr = data['xhr'] / 48440 / 0.0255
    yhr = data['yhr'] / 48440 / 0.0255

    print(np.max(xhr.flatten()))
    print(np.max(yhr.flatten()))


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
    im.append(ax39.contour(Qpiv[idx,:,:], levels=[0.99], origin="upper", colors='k', linewidths=1, alpha=0.5, extent=[0, xhr.max(), 0, yhr.max()]))
    im.append(ax40.imshow(hr_target[idx,:,:,0], origin="upper", extent=[0, xhr.max(), 0, yhr.max()], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax41.imshow(hr_predic[idx,:,:,0], origin="upper", extent=[0, xhr.max(), 0, yhr.max()], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax41.contour(Qgan[idx,:,:], levels=[0.99], origin="upper", colors='k', linewidths=1, alpha=0.5, extent=[0, xhr.max(), 0, yhr.max()]))
    im.append(ax42.imshow(dns_target[idx,:,:,0], origin="upper", extent=[0, xhr.max(), 0, yhr.max()], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax42.contour(Qdns[idx,:,:], levels=[0.99], origin="upper", colors='k', linewidths=1, alpha=0.5, extent=[0, xhr.max(), 0, yhr.max()]))

    im.append(ax43.imshow(lr_target[idx,:,:,1], origin="upper", extent=[0, xhr.max(), 0, yhr.max()], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax43.contour(Qpiv[idx,:,:], levels=[0.99], origin="upper", colors='k', linewidths=1, alpha=0.5, extent=[0, xhr.max(), 0, yhr.max()]))
    im.append(ax44.imshow(hr_target[idx,:,:,1], origin="upper", extent=[0, xhr.max(), 0, yhr.max()], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax45.imshow(hr_predic[idx,:,:,1], origin="upper", extent=[0, xhr.max(), 0, yhr.max()], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax45.contour(Qgan[idx,:,:], levels=[0.99], origin="upper", colors='k', linewidths=1, alpha=0.5, extent=[0, xhr.max(), 0, yhr.max()]))
    im.append(ax46.imshow(dns_target[idx,:,:,1], origin="upper", extent=[0, xhr.max(), 0, yhr.max()], cmap=cmap_v, vmin=-0.1, vmax=0.1))
    im.append(ax46.contour(Qdns[idx,:,:], levels=[0.99], origin="upper", colors='k', linewidths=1, alpha=0.5, extent=[0, xhr.max(), 0, yhr.max()]))

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


    ax47 = figure.add_axes([0.90, 0.170, 0.01, 0.1])
    ax48 = figure.add_axes([0.90, 0.035, 0.01, 0.1])
    ax47.tick_params(axis="both", direction="out", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=2)
    ax48.tick_params(axis="both", direction="out", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=2)
    
    cbar47 = figure.colorbar(im[31], cax=ax47, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-0.2, 0.0, 0.2])
    cbar48 = figure.colorbar(im[38], cax=ax48, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-0.1, 0, 0.1])
    cbar47.ax.set_title("$u/U_{\\infty}$")
    cbar48.ax.set_title("$v/U_{\\infty}$")
    [(t.set_horizontalalignment('right'), t.set_x(4)) for t in cbar47.ax.get_yticklabels()]
    [(t.set_horizontalalignment('right'), t.set_x(4)) for t in cbar48.ax.get_yticklabels()]
    # [(t.set_horizontalalignment('right'), t.set_x(3)) for t in cbar38.ax.get_yticklabels()]
    [(t.set_horizontalalignment('right'), t.set_x(4)) for t in cbar17.ax.get_yticklabels()]
    [(t.set_horizontalalignment('right'), t.set_x(4)) for t in cbar18.ax.get_yticklabels()]
    [(t.set_horizontalalignment('right'), t.set_x(4)) for t in cbar27.ax.get_yticklabels()]
    [(t.set_horizontalalignment('right'), t.set_x(4)) for t in cbar28.ax.get_yticklabels()]
    [(t.set_horizontalalignment('right'), t.set_x(3)) for t in cbar33.ax.get_yticklabels()]
    

    ax39.set_ylabel("$y/\\delta_{99}$", labelpad=2)
    ax43.set_xlabel("$x/\\delta_{99}$", labelpad=0)
    ax43.set_ylabel("$y/\\delta_{99}$", labelpad=2)
    ax44.set_xlabel("$x/\\delta_{99}$", labelpad=0)
    ax45.set_xlabel("$x/\\delta_{99}$", labelpad=0)
    ax46.set_xlabel("$x/\\delta_{99}$", labelpad=0)

    figure.savefig(f"../figs/lisboa01.pdf", dpi=1000)

    print('Hello')

    return


if __name__ == '__main__':

    main()