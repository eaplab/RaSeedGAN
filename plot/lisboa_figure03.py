# -*- coding: utf-8 -*-
"""
Created on Fri Sep 03 15:35:14 2021
@author: guemesturb
"""


import numpy as np
import matplotlib.pyplot as plt



def main():

    root_folder = f'/STORAGE01/aguemes/gan-piv/channel/ss{us:02}/' 

    for model_name in models_channel:
        
        res = 1/512/0.013  
        
        filename = f"{root_folder}results/predictions_{model_name}-noise-{noise:03d}.npz"

        Uptv_std = (np.expand_dims(np.load(f"{root_folder}tfrecords/scaling_us{us}_noise_{noise:03d}.npz")['Uptv_std'], axis=0)) *res#/ 512 / 0.013
        Vptv_std = (np.expand_dims(np.load(f"{root_folder}tfrecords/scaling_us{us}_noise_{noise:03d}.npz")['Vptv_std'], axis=0)) *res#/ 512 / 0.013
        
        data = np.load(filename)
        dns_target = data['dns_target'] * res
        hr_predic  = data['hr_predic'] 

        error_u = np.sqrt(np.nanmean(np.divide((dns_target[:, :, :, 0] -  hr_predic[:, :, :, 0])**2, Uptv_std * Uptv_std, out=np.zeros_like(dns_target[:, :, :, 0]), where=Uptv_std!=0)))
        error_v = np.sqrt(np.nanmean(np.divide((dns_target[:, :, :, 1] -  hr_predic[:, :, :, 1])**2, Vptv_std * Vptv_std, out=np.zeros_like(dns_target[:, :, :, 0]), where=Vptv_std!=0)))
        
        error_mse_u_channel.append(
            error_u
        )
        
        error_mse_v_channel.append(
            error_v
        )

    root_folder = f'/STORAGE01/aguemes/gan-piv/pinball/ss{us:02}/' 

    for model_name in models_pinball:
        
        res = 1
        
        filename = f"{root_folder}results/predictions_{model_name}-noise-{noise:03d}.npz"

        Uptv_std = (np.expand_dims(np.load(f"{root_folder}tfrecords/scaling_us{us}_noise_{noise:03d}.npz")['Uptv_std'], axis=0)) *res#/ 512 / 0.013
        Vptv_std = (np.expand_dims(np.load(f"{root_folder}tfrecords/scaling_us{us}_noise_{noise:03d}.npz")['Vptv_std'], axis=0)) *res#/ 512 / 0.013
        
        data = np.load(filename)
        dns_target = data['dns_target'] * res
        hr_predic  = data['hr_predic'] 

        error_u = np.sqrt(np.nanmean(np.divide((dns_target[:, :, :, 0] -  hr_predic[:, :, :, 0])**2, Uptv_std * Uptv_std, out=np.zeros_like(dns_target[:, :, :, 0]), where=Uptv_std!=0)))
        error_v = np.sqrt(np.nanmean(np.divide((dns_target[:, :, :, 1] -  hr_predic[:, :, :, 1])**2, Vptv_std * Vptv_std, out=np.zeros_like(dns_target[:, :, :, 0]), where=Vptv_std!=0)))
        
        error_mse_u_pinball.append(
            error_u
        )
        
        error_mse_v_pinball.append(
            error_v
        )

    plt.rc('text', usetex='True')
    plt.rc('font', family='Serif', size='6')

    fig_width_pt = 510
    inches_per_pt = 1.0 / 72.27  
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width * 0.28 

    figure = plt.figure('panel01', figsize=(fig_width, fig_height))
    im = []

    ax01 = figure.add_axes([0.100, 0.18, 0.35, 0.67])
    ax02 = figure.add_axes([0.550, 0.18, 0.35, 0.67])

    import matplotlib
    cmap = matplotlib.cm.get_cmap('inferno')

    c1 = cmap(0.7)
    c2 = cmap(1.0)

    ax02.plot(n_samples_channel, error_mse_u_channel, color=c1, label='$\\varepsilon_u$')
    ax02.plot(n_samples_channel, error_mse_v_channel, color=c2, label='$\\varepsilon_v$')
    ax01.plot(n_samples_pinball, error_mse_u_pinball, color=c1)
    ax01.plot(n_samples_pinball, error_mse_v_pinball, color=c2)
    ax01.set_title('Test Case 1')
    ax02.set_title('Test Case 2')
    ax01.set_xlim([0,4000])
    ax02.set_xlim([0,9000])
    ax01.set_ylim([0.15,0.35])
    ax02.set_ylim([0.25,0.35])
    ax01.set_xlabel("Number of training samples")
    ax02.set_xlabel("Number of training samples")
    ax01.set_ylabel("$\\varepsilon$")
    ax02.set_ylabel("$\\varepsilon$")
    ax02.legend()

    figure.savefig(f"../figs/lisboa03.pdf", dpi=1000)


    # plot_nsamples_analysis(n_samples_pinball, error_mse_u_pinball, error_mse_v_pinball, n_samples_channel, error_mse_u_channel, error_mse_v_channel)
    

    return


if __name__ == '__main__':

    """
        Parsing arguments
    """

    us = 4
    noise = 10

    models_channel = ['architecture-01', 'architecture-02', 'architecture-03', 'architecture-04', 'architecture-05', 'architecture-06', 'architecture-07', 'architecture-08', 'architecture-09', 'architecture-10', 'architecture-11']
    models_pinball = ['architecture-01', 'architecture-02', 'architecture-03', 'architecture-04', 'architecture-05', 'architecture-06', 'architecture-07', 'architecture-08', 'architecture-09', 'architecture-10', 'architecture-11']
    n_samples_channel = [8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 750, 500, 250]
    n_samples_pinball = [3200, 2800, 2400, 2000, 1600, 1200, 800, 400, 300, 200, 100]
    error_mse_u_channel = []
    error_mse_v_channel = []
    error_mse_u_pinball = []
    error_mse_v_pinball = []

    main()