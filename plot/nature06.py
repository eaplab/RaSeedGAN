import os
import re
import matplotlib
import numpy as np
import scipy.io as sio
import TheArtist
import matplotlib.pyplot as plt


def main():
    plt.rc('text', usetex='True')
    plt.rc('font', family='Serif', size='6')

    fig_width_pt = 510
    inches_per_pt = 1.0 / 72.27
    ratio = 2 / (1 + 5 ** 0.5)   
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width * ratio / 4 * 4 +0.3*2

    figure = plt.figure('panel15', figsize=(fig_width, fig_height))

    ax01 = figure.add_axes([0.040, 0.94, 0.42, 0.05])
    ax02 = figure.add_axes([0.520, 0.94, 0.42, 0.05])

    ax01.text(0.5, 0.5, 'Test Case 2', horizontalalignment='center', verticalalignment='center', transform=ax01.transAxes, fontsize=8)
    ax01.axis("off")
    ax02.text(0.5, 0.5, 'Test Case 4', horizontalalignment='center', verticalalignment='center', transform=ax02.transAxes, fontsize=8)
    ax02.axis("off")

    axa = figure.add_axes([0.010, 0.92, 0.12, 0.05])
    axb = figure.add_axes([0.010, 0.65, 0.12, 0.05])

    axa.text(0.05, 0.5, 'a)', horizontalalignment='center', verticalalignment='center', transform=axa.transAxes, fontsize=8)
    axa.axis("off")
    axb.text(0.05, 0.5, 'b)', horizontalalignment='center', verticalalignment='center', transform=axb.transAxes, fontsize=8)
    axb.axis("off")

    ax03 = figure.add_axes([0.040, 0.72, 0.2, 0.2])
    ax04 = figure.add_axes([0.280, 0.72, 0.2, 0.2])
    ax05 = figure.add_axes([0.520, 0.72, 0.2, 0.2])
    ax06 = figure.add_axes([0.760, 0.72, 0.2, 0.2])

    ax07 = figure.add_axes([0.060, 0.47, 0.40, 0.17])
    ax08 = figure.add_axes([0.540, 0.47, 0.40, 0.17])
    ax09 = figure.add_axes([0.060, 0.27, 0.40, 0.17])
    ax10 = figure.add_axes([0.540, 0.27, 0.40, 0.17])
    ax11 = figure.add_axes([0.060, 0.07, 0.40, 0.17])
    ax12 = figure.add_axes([0.540, 0.07, 0.40, 0.17])

    """
        Channel
    """

    filename = f"/storage/aguemes/gan-piv/channel/ss08/results/predictions_architecture-01-noise-010.npz"
    data = np.load(filename)
    dns_target = data['dns_target'] / 512 / 0.013
    hr_predic  = data['hr_predic'] 
    lr_target  = data['lr_target'] 
    fl_target  = data['fl_target']
    cbc_predic = data['cbc_predic'] / 512 / 0.013 
    gap_predic = data['gap_predic'] / 512 / 0.013 

    xhr = data['xhr']
    yhr = data['yhr'][:,0]
    grid_path = f"/storage/aguemes/gan-piv/channel/ss01/piv_noise010/SS1_grid.mat"
    grid = sio.loadmat(grid_path)
    xlr = np.array(grid['X']).T
    ylr = np.array(grid['Y']).T[0,:]
    yplus = yhr / 512 * 0.0499 / 0.00005
    yppiv = ylr / 512 * 0.0499 / 0.00005
    
    cmapR = matplotlib.cm.get_cmap('Reds')
    cmapB = matplotlib.cm.get_cmap('Blues')
    cmapY = matplotlib.cm.get_cmap('plasma')
    cmapV = matplotlib.cm.get_cmap('Purples')
    cmapG = matplotlib.cm.get_cmap('Greens')

    cpiv = cmapR(0.6)
    cgan = cmapB(1.0)
    cgan4 = cmapB(0.6)
    cgan2 = cmapB(0.2)
    ccbc = cmapY(0.9)
    cgap = cmapV(0.9)

    cy1 = cmapG(1.0)
    cy2 = cmapG(0.7)
    cy3 = cmapG(0.4)

    ax03.plot([50, 50], [0, 30], color=cy1, linewidth=1, linestyle='-')
    ax03.plot([250, 250], [0, 30], color=cy2, linewidth=1, linestyle='-')
    ax03.plot([500, 500], [0, 30], color=cy3, linewidth=1, linestyle='-')

    ax04.plot([50, 50], [0, 25], color=cy1, linewidth=1, linestyle='-')
    ax04.plot([250, 250], [0, 25], color=cy2, linewidth=1, linestyle='-')
    ax04.plot([500, 500], [0, 25], color=cy3, linewidth=1, linestyle='-')


    mean_u_gan = np.mean(np.mean(hr_predic[:, ::-1, :, 0], axis=0), axis=1) / 0.0499
    mean_u_piv = np.mean(np.mean(lr_target[:, ::-1, :, 0], axis=0), axis=1) / 0.0499
    mean_u_cbc = np.mean(np.nanmean(cbc_predic[:, ::-1, :, 0], axis=0), axis=1) / 0.0499 
    mean_u_gap = np.mean(np.nanmean(gap_predic[:, ::-1, :, 0], axis=0), axis=1) / 0.0499 

    mean_uu_gan = np.mean(np.var(hr_predic[:, ::-1, :, 0], axis=0), axis=1) / 0.0499 / 0.0499
    mean_uu_piv = np.mean(np.var(lr_target[:, ::-1, :, 0], axis=0), axis=1) / 0.0499 / 0.0499
    mean_uu_cbc = np.mean(np.nanvar(cbc_predic[:, ::-1, :, 0], axis=0, ddof=-1), axis=1) / 0.0499 / 0.0499
    mean_uu_gap = np.mean(np.nanvar(gap_predic[:, ::-1, :, 0], axis=0, ddof=-1), axis=1) / 0.0499 / 0.0499
    
    filename = 're1000_full_stat.mat'
    data = sio.loadmat(filename)
    mean_u_dns = data['u']
    mean_uu_dns = data['uu']
    ydns = data['y']
    
    ax03.semilogx(ydns[0,:], mean_u_dns[0,:], color='k', linewidth=2, linestyle='-', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    ax03.semilogx(yplus, mean_u_cbc, color=ccbc, linewidth=1, linestyle='-.', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    ax03.semilogx(yplus, mean_u_gap, color=cgap, linewidth=1, linestyle='-.', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    ax03.semilogx(yplus, mean_u_gan, color=cgan, linewidth=1, linestyle='--', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=1)
    ax03.semilogx(yppiv, mean_u_piv, color=cpiv, linewidth=1, linestyle=':', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)

    ax04.semilogx(ydns[0,:], mean_uu_dns[0,:], color='k', linewidth=2, linestyle='-', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    ax04.semilogx(yplus, mean_uu_cbc, color=ccbc, linewidth=1, linestyle='-.', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    ax04.semilogx(yplus, mean_uu_gap, color=cgap, linewidth=1, linestyle='-.', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    ax04.semilogx(yplus, mean_uu_gan, color=cgan, linewidth=1, linestyle='--', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    ax04.semilogx(yppiv, mean_uu_piv, color=cpiv, linewidth=1, linestyle=':', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)

    data = sio.loadmat('channel_spectra.mat')
    print(data.keys())
    puustar = (data['lstar']*data['utau']*data['utau'])[0][0]
    ax07.loglog(data['freq'][0]*data['lstar'][0], data['puuX_DNS'][:,data['idx_aux'][0,0]-1]/puustar, color='k')
    ax07.loglog(data['freq'][0]*data['lstar'][0], data['puuX_CUBIC'][:,data['idx_aux'][0,0]-1]/puustar, color=ccbc, linewidth=1, linestyle='-.')
    ax07.loglog(data['freq'][0]*data['lstar'][0], data['puuX_GAP'][:,data['idx_aux'][0,0]-1]/puustar, color=cgap, linewidth=1, linestyle='-.')
    ax07.loglog(data['freq'][0]*data['lstar'][0], data['puuX_GAN'][:,data['idx_aux'][0,0]-1]/puustar, color=cgan, linewidth=1, linestyle='--')
    ax07.loglog(data['freq4'][0]*data['lstar'][0], data['puuX_GAN4'][:,data['idx4_aux'][0,0]-1]/puustar, color=cgan4, linewidth=1, linestyle='--')
    ax07.loglog(data['freq2'][0]*data['lstar'][0], data['puuX_GAN2'][:,data['idx2_aux'][0,0]-1]/puustar, color=cgan2, linewidth=1, linestyle='--')
    ax07.loglog(data['freq_LR'][0]*data['lstar'][0], data['puuX_LR'][:,data['idx_LR_aux'][0,0]-1]/puustar, color=cpiv, linewidth=1, linestyle=':')
    
    ax09.loglog(data['freq'][0]*data['lstar'][0], data['puuX_DNS'][:,data['idx_aux'][0,1]-1]/puustar, color='k')
    ax09.loglog(data['freq'][0]*data['lstar'][0], data['puuX_CUBIC'][:,data['idx_aux'][0,1]-1]/puustar, color=ccbc, linewidth=1, linestyle='-.')
    ax09.loglog(data['freq'][0]*data['lstar'][0], data['puuX_GAP'][:,data['idx_aux'][0,1]-1]/puustar, color=cgap, linewidth=1, linestyle='-.')
    ax09.loglog(data['freq'][0]*data['lstar'][0], data['puuX_GAN'][:,data['idx_aux'][0,1]-1]/puustar, color=cgan, linewidth=1, linestyle='--')
    ax09.loglog(data['freq4'][0]*data['lstar'][0], data['puuX_GAN4'][:,data['idx4_aux'][0,1]-1]/puustar, color=cgan4, linewidth=1, linestyle='--')
    ax09.loglog(data['freq2'][0]*data['lstar'][0], data['puuX_GAN2'][:,data['idx2_aux'][0,1]-1]/puustar, color=cgan2, linewidth=1, linestyle='--')
    ax09.loglog(data['freq_LR'][0]*data['lstar'][0], data['puuX_LR'][:,data['idx_LR_aux'][0,1]-1]/puustar, color=cpiv, linewidth=1, linestyle=':')
    ax11.loglog(data['freq'][0]*data['lstar'][0], data['puuX_DNS'][:,data['idx_aux'][0,2]-1]/puustar, color='k')
    ax11.loglog(data['freq'][0]*data['lstar'][0], data['puuX_CUBIC'][:,data['idx_aux'][0,2]-1]/puustar, color=ccbc, linewidth=1, linestyle='-.')
    ax11.loglog(data['freq'][0]*data['lstar'][0], data['puuX_GAP'][:,data['idx_aux'][0,2]-1]/puustar, color=cgap, linewidth=1, linestyle='-.')
    ax11.loglog(data['freq'][0]*data['lstar'][0], data['puuX_GAN'][:,data['idx_aux'][0,2]-1]/puustar, color=cgan, linewidth=1, linestyle='--')
    ax11.loglog(data['freq4'][0]*data['lstar'][0], data['puuX_GAN4'][:,data['idx4_aux'][0,2]-1]/puustar, color=cgan4, linewidth=1, linestyle='--')
    ax11.loglog(data['freq2'][0]*data['lstar'][0], data['puuX_GAN2'][:,data['idx2_aux'][0,2]-1]/puustar, color=cgan2, linewidth=1, linestyle='--')
    ax11.loglog(data['freq_LR'][0]*data['lstar'][0], data['puuX_LR'][:,data['idx_LR_aux'][0,2]-1]/puustar, color=cpiv, linewidth=1, linestyle=':')
    
    ax07.text(0.8,0.8,"$y^+=50$", transform=ax07.transAxes, color=cy1)
    ax09.text(0.8,0.8,"$y^+=250$", transform=ax09.transAxes, color=cy2)
    ax11.text(0.8,0.8,"$y^+=500$", transform=ax11.transAxes, color=cy3)
    # """
    #     TBL
    # """

    filename = f"/storage/aguemes/gan-piv/exptbl/ss04/results/predictions_architecture-01-noise-000.npz"
    data = np.load(filename)
    hr_predic  = data['hr_predic'] 
    lr_target  = data['lr_target'] 
    fl_target  = data['fl_target']
    cbc_predic = data['cbc_predic']
    gap_predic = data['gap_predic']
    dns_target = data['dns_target']
    yhr = data['yhr'][:,0]
    ylr = data['ylr'][:,0]

    filename = 'PTV_Guemes_stat.mat'
    data = sio.loadmat(filename)
    utau = data['OutData'][0][0][3][0][0]
    nu = data['OutData'][0][0][4][0][0]
    mean_u_dns = np.mean(np.mean(dns_target[:, ::-1, :, 0], axis=0), axis=1) / (48440 * 0.000017 * utau)
    mean_uu_dns = np.mean(np.var(dns_target[:, ::-1, :, 0], axis=0), axis=1) / (48440 * 0.000017 * utau)**2
    

    mean_u_piv = np.mean(lr_target[:, ::-1, :, 0], axis=(0, 2)) / 48440 / 0.000017 / utau
    mean_uu_piv = np.mean(np.nanvar(lr_target[:, ::-1, :, 0], axis=0), axis=1) / (48440 * 0.000017 * utau)**2

    mean_u_gan = np.mean(hr_predic[:, ::-1, :, 0], axis=(0, 2)) / 48440 / 0.000017 / utau
    mean_u_cbc = np.nanmean(cbc_predic[:, ::-1, :, 0], axis=(0, 2)) / 48440 / 0.000017 / utau
    mean_u_gap = np.nanmean(gap_predic[:, ::-1, :, 0], axis=(0, 2)) / 48440 / 0.000017 / utau
    mean_uu_gan = np.mean(np.nanvar(hr_predic[:, ::-1, :, 0], axis=0), axis=1) / (48440 * 0.000017 * utau)**2
    mean_uu_cbc = np.mean(np.nanvar(cbc_predic[:, ::-1, :, 0], axis=0), axis=1) / (48440 * 0.000017 * utau)**2
    mean_uu_gap = np.mean(np.nanvar(gap_predic[:, ::-1, :, 0], axis=0), axis=1) / (48440 * 0.000017 * utau)**2

    yppiv = ylr / 48440 * utau / nu 
    yplus = yhr / 48440 * utau / nu + (data['OutData'][0][0][0][0] - yhr[0] / 48440) * utau / nu
    

    ax05.plot([250, 250], [0, 30], color=cy1, linewidth=1, linestyle='-')
    ax05.plot([500, 500], [0, 30], color=cy2, linewidth=1, linestyle='-')
    ax05.plot([800, 800], [0, 30], color=cy3, linewidth=1, linestyle='-')

    ax06.plot([250, 250], [0, 30], color=cy1, linewidth=1, linestyle='-')
    ax06.plot([500, 500], [0, 30], color=cy2, linewidth=1, linestyle='-')
    ax06.plot([800, 800], [0, 30], color=cy3, linewidth=1, linestyle='-')


    data = sio.loadmat('computed_vel_3270_dns.prof.mat')
    
    ax05.semilogx(yplus, mean_u_dns, color='k', linewidth=2, linestyle='-', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    ax05.semilogx(data['yplus'], data['Uplus'], color='dimgrey', linewidth=2, linestyle='-', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    ax05.semilogx(yplus, mean_u_cbc, color=ccbc, linewidth=1, linestyle='-.', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=3)
    ax05.semilogx(yplus, mean_u_gap, color=cgap, linewidth=1, linestyle='-.', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=3)
    ax05.semilogx(yplus, mean_u_gan, color=cgan4, linewidth=1, linestyle='--', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=3)
    ax05.semilogx(yppiv, mean_u_piv, color=cpiv, linewidth=1, linestyle=':', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=3)


    ax06.semilogx(yplus, mean_uu_dns, color='k', linewidth=2, linestyle='-', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    ax06.semilogx(data['yplus'], data['urmsplus']**2, color='dimgrey', linewidth=2, linestyle='-', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    ax06.semilogx(yplus, mean_uu_cbc, color=ccbc, linewidth=1, linestyle='-.', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    ax06.semilogx(yplus, mean_uu_gap, color=cgap, linewidth=1, linestyle='-.', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    ax06.semilogx(yplus, mean_uu_gan, color=cgan4, linewidth=1, linestyle='--', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    ax06.semilogx(yppiv, mean_uu_piv, color=cpiv, linewidth=1, linestyle=':', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)

    data = sio.loadmat('exp_spectra.mat')
    puustar = (data['lstar']*data['utau']*data['utau'])[0][0]
    ax08.loglog(data['freq'][0]*data['lstar'][0], data['puuX_DNS'][:,data['idx_aux'][0,0]-1]/puustar, color='k')
    ax08.loglog(data['freq'][0]*data['lstar'][0], data['puuX_CUBIC'][:,data['idx_aux'][0,0]-1]/puustar, color=ccbc, linewidth=1, linestyle='-.')
    ax08.loglog(data['freq'][0]*data['lstar'][0], data['puuX_GAP'][:,data['idx_aux'][0,0]-1]/puustar, color=cgap, linewidth=1, linestyle='-.')
    ax08.loglog(data['freq'][0]*data['lstar'][0], data['puuX_GAN'][:,data['idx_aux'][0,0]-1]/puustar, color=cgan4, linewidth=1, linestyle='--')
    ax08.loglog(data['freq_LR'][0]*data['lstar'][0], data['puuX_LR'][:,data['idx_LR_aux'][0,0]-1]/puustar, color=cpiv, linewidth=1, linestyle=':')
    ax10.loglog(data['freq'][0]*data['lstar'][0], data['puuX_DNS'][:,data['idx_aux'][0,1]-1]/puustar, color='k')
    ax10.loglog(data['freq'][0]*data['lstar'][0], data['puuX_CUBIC'][:,data['idx_aux'][0,1]-1]/puustar, color=ccbc, linewidth=1, linestyle='-.')
    ax10.loglog(data['freq'][0]*data['lstar'][0], data['puuX_GAP'][:,data['idx_aux'][0,1]-1]/puustar, color=cgap, linewidth=1, linestyle='-.')
    ax10.loglog(data['freq'][0]*data['lstar'][0], data['puuX_GAN'][:,data['idx_aux'][0,1]-1]/puustar, color=cgan4, linewidth=1, linestyle='--')
    ax10.loglog(data['freq_LR'][0]*data['lstar'][0], data['puuX_LR'][:,data['idx_LR_aux'][0,1]-1]/puustar, color=cpiv, linewidth=1, linestyle=':')
    ax12.loglog(data['freq'][0]*data['lstar'][0], data['puuX_DNS'][:,data['idx_aux'][0,2]-1]/puustar, color='k')
    ax12.loglog(data['freq'][0]*data['lstar'][0], data['puuX_CUBIC'][:,data['idx_aux'][0,2]-1]/puustar, color=ccbc, linewidth=1, linestyle='-.')
    ax12.loglog(data['freq'][0]*data['lstar'][0], data['puuX_GAP'][:,data['idx_aux'][0,2]-1]/puustar, color=cgap, linewidth=1, linestyle='-.')
    ax12.loglog(data['freq'][0]*data['lstar'][0], data['puuX_GAN'][:,data['idx_aux'][0,2]-1]/puustar, color=cgan4, linewidth=1, linestyle='--')
    ax12.loglog(data['freq_LR'][0]*data['lstar'][0], data['puuX_LR'][:,data['idx_LR_aux'][0,2]-1]/puustar, color=cpiv, linewidth=1, linestyle=':')
    
    

    ax08.text(0.8,0.8,"$y^+=250$", transform=ax08.transAxes, color=cy1)
    ax10.text(0.8,0.8,"$y^+=500$", transform=ax10.transAxes, color=cy2)
    ax12.text(0.8,0.8,"$y^+=800$", transform=ax12.transAxes, color=cy3)

    """
        Settings
    """

    ax03.set_ylabel("$U^+$", labelpad=2)
    ax04.set_ylabel("$u^{\\prime}u^{\\prime ^+}$", labelpad=-1)
    ax05.set_xlabel("$y^+$", labelpad=2)
    ax05.set_ylabel("$U^+$", labelpad=2)
    ax06.set_xlabel("$y^+$", labelpad=2)
    ax04.set_xlabel("$y^+$", labelpad=2)
    ax03.set_xlabel("$y^+$", labelpad=2)
    ax06.set_ylabel("$u^{\\prime}u^{\\prime ^+}$", labelpad=-1)
    ax11.set_xlabel("$f_x^+$", labelpad=2)
    ax12.set_xlabel("$f_x^+$", labelpad=2)
    # ax08.set_xlabel("$f^+$", labelpad=2)
    # ax09.set_xlabel("$f^+$", labelpad=2)
    # ax10.set_xlabel("$f^+$", labelpad=2)
    ax07.set_ylabel("$\\phi_{uu_x}$", labelpad=0)
    ax09.set_ylabel("$\\phi_{uu_x}$", labelpad=0)
    ax11.set_ylabel("$\\phi_{uu_x}$", labelpad=0)
    ax08.set_ylabel("$\\phi_{uu_x}$", labelpad=0)
    ax10.set_ylabel("$\\phi_{uu_x}$", labelpad=0)
    ax12.set_ylabel("$\\phi_{uu_x}$", labelpad=0)


    ax03.set_xlim([1, 2000])
    ax04.set_xlim([1, 2000])
    ax05.set_xlim([1, 2000])
    ax06.set_xlim([1, 2000])
    ax03.set_ylim([0, 30])
    ax04.set_ylim([0, 10])
    ax05.set_ylim([0, 30])
    ax06.set_ylim([0, 10])
    ax07.set_ylim([0.005, 1100])
    ax08.set_ylim([0.005, 1100])
    ax09.set_ylim([0.005, 1100])
    ax10.set_ylim([0.005, 1100])
    ax11.set_ylim([0.005, 1100])
    ax12.set_ylim([0.005, 1100])

    ax04.set_yticks([0, 1, 2, 3, 4])
    ax04.set_yticks([0,  2, 4, 6, 8, 10])
    ax06.set_yticks([0,  2, 4, 6, 8, 10])

    ax08.set_xticks([])
    ax07.set_xticks([])
    ax09.set_xticks([])
    ax10.set_xticks([])

    ax03.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)
    ax04.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)
    ax05.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)
    ax06.tick_params(axis="both", direction="in", which="both", pad=2, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)
    ax07.tick_params(axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)
    ax08.tick_params(axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)
    ax09.tick_params(axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)
    ax10.tick_params(axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)
    ax11.tick_params(axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)
    ax12.tick_params(axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)

    axins = ax06.inset_axes([0.65, 0.65, 0.3, 0.3])
    data = sio.loadmat('computed_vel_3270_dns.prof.mat')
    axins.semilogx(yplus, mean_uu_dns, color='k', linewidth=2, linestyle='-', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    axins.semilogx(data['yplus'], data['urmsplus']**2, color='dimgrey', linewidth=2, linestyle='-', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    axins.semilogx(yplus, mean_uu_cbc, color=ccbc, linewidth=1, linestyle='-.', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    axins.semilogx(yplus, mean_uu_gan, color=cgan4, linewidth=1, linestyle='--', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    axins.semilogx(yppiv, mean_uu_piv, color=cpiv, linewidth=1, linestyle=':', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    axins.set_xlim([1000, np.max(yplus)])
    axins.set_ylim([0,1])
    axins.set_xticklabels([],minor=True)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.tick_params(axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)
    ax06.indicate_inset_zoom(axins, edgecolor="black", alpha=1)
    figure.savefig(f"../figs/nature06.pdf", dpi=1000)

    return


if __name__ == '__main__':

    main()