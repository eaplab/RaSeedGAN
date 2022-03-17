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
    fig_height = fig_width * 0.30 

    figure = plt.figure('panel01', figsize=(fig_width, fig_height))
    im = []


    tx01 = figure.add_axes([0.010, 0.8, 0.21, 0.05])
    tx01.text(0.0, 0.5, 'a) In-Training Quantities', fontsize=8)
    tx01.axis("off")
    tx02 = figure.add_axes([0.010, 0.32, 0.21, 0.05])
    tx02.text(0.0, 0.5, 'b) Out-Of-Training Quantities', fontsize=8)
    tx02.axis("off")

    ax01 = figure.add_axes([0.040, 0.9, 0.21, 0.05])
    ax02 = figure.add_axes([0.260, 0.9, 0.21, 0.05])
    ax03 = figure.add_axes([0.480, 0.9, 0.21, 0.05])
    ax04 = figure.add_axes([0.700, 0.9, 0.21, 0.05])

    ax01.text(0.5, 0.5, 'LR Input', horizontalalignment='center', verticalalignment='center', transform=ax01.transAxes, fontsize=8)
    ax01.axis("off")
    ax02.text(0.5, 0.5, 'RaSeedGAN', horizontalalignment='center', verticalalignment='center', transform=ax02.transAxes, fontsize=8)
    ax02.axis("off")
    ax03.text(0.5, 0.5, 'Complete HR Reference', horizontalalignment='center', verticalalignment='center', transform=ax03.transAxes, fontsize=8)
    ax03.axis("off")
    ax04.text(0.5, 0.5, 'Cubic Interpolation', horizontalalignment='center', verticalalignment='center', transform=ax04.transAxes, fontsize=8)
    ax04.axis("off")


    ax09 = figure.add_axes([0.040, 0.06, 0.21, 0.27])
    ax11 = figure.add_axes([0.260, 0.06, 0.21, 0.27])
    ax12 = figure.add_axes([0.480, 0.06, 0.21, 0.27])
    ax13 = figure.add_axes([0.700, 0.06, 0.21, 0.27])

    filename = f"/STORAGE01/aguemes/gan-piv/pinball/ss08/results/predictions_architecture-01-noise-010.npz"
    data = np.load(filename)
    dns_target = data['dns_target'] 
    hr_predic  = data['hr_predic'] 
    hr_target  = data['hr_target'] 
    lr_target  = data['lr_target'] 
    cbc_predic  = data['cbc_predic'] 
    fl_target  = data['fl_target']
    hr_target[np.repeat(fl_target, 2, axis=3) == 0] = np.nan
    xlr = data['xlr'] 
    xhr = data['xhr'] 
    ylr = data['ylr'] 
    yhr = data['yhr'] 

    Res = 25
    xmin = -5
    ymin = -4 + 4 / Res
    xhr = xhr / Res + xmin
    xlr = xlr / Res + xmin
    yhr = yhr / Res + ymin
    dudx, dudy = np.gradient(dns_target[:,:,:,0], 0.07999992, 0.07999992, axis=[1,2], edge_order=2)
    dvdx, dvdy = np.gradient(dns_target[:,:,:,1], 0.07999992, 0.07999992, axis=[1,2], edge_order=2)
    wdns = dvdx - dudy
    
    dudx, dudy = np.gradient(lr_target[:,:,:,0], 0.63999987, 0.63999987, axis=[1,2], edge_order=2)
    dvdx, dvdy = np.gradient(lr_target[:,:,:,1], 0.63999987, 0.63999987, axis=[1,2], edge_order=2)
    wpiv = dvdx - dudy
    
    dudx, dudy = np.gradient(hr_predic[:,:,:,0], 0.07999992, 0.07999992, axis=[1,2], edge_order=2)
    dvdx, dvdy = np.gradient(hr_predic[:,:,:,1], 0.07999992, 0.07999992, axis=[1,2], edge_order=2)
    wgan = dvdx - dudy
    
    dudx, dudy = np.gradient(cbc_predic[:,:,:,0], 0.07999992, 0.07999992, axis=[1,2], edge_order=2)
    dvdx, dvdy = np.gradient(cbc_predic[:,:,:,1], 0.07999992, 0.07999992, axis=[1,2], edge_order=2)
    wcbc = dvdx - dudy

    cmap = matplotlib.cm.get_cmap("RdBu_r").copy()
    cmap.set_bad(color='k')

    im.append(ax09.imshow(wpiv[0,:,:], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap, vmin=-1.5, vmax=1.5))
    im.append(ax11.imshow(wgan[0,:,:], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap, vmin=-1.5, vmax=1.5))
    im.append(ax12.imshow(wdns[0,:,:], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap, vmin=-1.5, vmax=1.5))
    im.append(ax13.imshow(wcbc[0,:,:], origin="upper", extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap, vmin=-1.5, vmax=1.5))

    draw_circle1 = plt.Circle((- 1.299, 0), 0.5, edgecolor=None, facecolor='gray')
    draw_circle2 = plt.Circle((0, 0.75), 0.5,    edgecolor=None, facecolor='gray')
    draw_circle3 = plt.Circle((0, -0.75), 0.5,   edgecolor=None, facecolor='gray')
    ax09.add_artist(draw_circle1)
    ax09.add_artist(draw_circle2)
    ax09.add_artist(draw_circle3)
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

    ax09.set_xticks([-5, 0, 5, 10, 15])
    ax09.set_yticks([-3, 0, 3])
    ax11.set_xticks([-5, 0, 5, 10, 15])
    ax11.set_yticks([])
    ax12.set_xticks([-5, 0, 5, 10, 15])
    ax12.set_yticks([])
    ax13.set_xticks([-5, 0, 5, 10, 15])
    ax13.set_yticks([])

    ax09.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax11.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax12.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax13.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)

    ax17 = figure.add_axes([0.95, 0.08, 0.01, 0.2])
    ax17.tick_params(axis="both", direction="out", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=2)

    cbar17 = figure.colorbar(im[0], cax=ax17, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-1.5, 0.0, 1.5])
    cbar17.ax.set_title("$\\omega D/U_{\\infty}$")

    ax09.set_ylabel("$y/D$", labelpad=0)
    ax09.set_xlabel("$x/D$", labelpad=0)
    ax11.set_xlabel("$x/D$", labelpad=0)
    ax12.set_xlabel("$x/D$", labelpad=0)
    ax13.set_xlabel("$x/D$", labelpad=0)


    """
        Channel
    """

    ax18 = figure.add_axes([0.040, 0.50, 0.21, 0.27])
    ax19 = figure.add_axes([0.260, 0.50, 0.21, 0.27])
    ax20 = figure.add_axes([0.480, 0.50, 0.21, 0.27])
    ax21 = figure.add_axes([0.700, 0.50, 0.21, 0.27])

    filename = f"/STORAGE01/aguemes/gan-piv/channel/ss08/results/predictions_architecture-01-noise-010.npz"
    data = np.load(filename)
    dns_target = data['dns_target'] * 1/512/0.013
    hr_predic  = data['hr_predic'] 
    cbc_predic  = data['cbc_predic'] * 1/512/0.013
    hr_target  = data['hr_target'] 
    lr_target  = data['lr_target'] 
    fl_target  = data['fl_target']
    hr_target[np.repeat(fl_target, 2, axis=3) == 0] = np.nan
    cmap_u = matplotlib.cm.get_cmap("RdGy").copy()
    cmap_u.set_bad(color='k')


    dns_target = dns_target - np.mean(dns_target, axis=(0,))
    lr_target = lr_target - np.mean(lr_target, axis=(0,))
    hr_target = hr_target - np.nanmean(hr_target, axis=(0,))
    hr_predic = hr_predic - np.mean(hr_predic, axis=(0,))
    cbc_predic = cbc_predic - np.mean(cbc_predic, axis=(0,))

    im.append(ax18.imshow(lr_target[90,:,:,0], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax19.imshow(hr_predic[90,:,:,0], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax20.imshow(dns_target[90,:,:,0], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    im.append(ax21.imshow(cbc_predic[90,:,:,0], origin="upper", extent=[0, 2, 0, 1], cmap=cmap_u, vmin=-0.2, vmax=0.2))
    
    ax18.set_xticks([0, 1, 2])
    ax18.set_yticks([0, 0.5, 1])
    ax19.set_xticks([0, 1, 2])
    ax19.set_yticks([])
    ax20.set_xticks([0, 1, 2])
    ax20.set_yticks([])
    ax21.set_xticks([0, 1, 2])
    ax21.set_yticks([])

    ax18.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax19.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax20.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    ax21.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)

    ax18.set_ylabel("$y/h$", labelpad=0.5)
    ax18.set_xlabel("$x/h$", labelpad=0)
    ax19.set_xlabel("$x/h$", labelpad=0)
    ax20.set_xlabel("$x/h$", labelpad=0)
    ax21.set_xlabel("$x/h$", labelpad=0)

    ax27 = figure.add_axes([0.95, 0.5, 0.01, 0.2])
    ax27.tick_params(axis="both", direction="out", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=2)

    cbar27 = figure.colorbar(im[-1], cax=ax27, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-0.2, 0.0, 0.2])
    cbar27.ax.set_title("$u/U_b$")

    figure.savefig(f"../figs/panel16.pdf", dpi=1000)

    print('Hello')

    return


if __name__ == '__main__':

    main()