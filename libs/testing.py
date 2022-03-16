# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:04:07 2021
@author: guemesturb
"""


import os
import re
import sys
sys.path.insert(1, '/home/aguemes/tools/TheArtist')
import matplotlib
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tqdm import tqdm
from artist import TheArtist
import matplotlib.pyplot as plt


def compute_predictions(root_folder, model_name, dataset_test, n_samp, nx, ny, res, us, channels, noise, generator, discriminator, generator_optimizer, discriminator_optimizer, save_mat=False):
    """
        Training logic for GAN architectures.

    :param root_folder:             String containg the folder where data is stored
    :param model_name:              String containing the assigned name to the model, for storage purposes.
    :param dataset_test:            Tensorflow pipeline for the testing dataset.
    :param n_samp:                  Integer indicating the number of samples in the testing dataset
    :param nx:                      Integer indicating the grid points in the streamwise direction for the low-resolution data.
    :param ny:                      Integer indicating the grid points in the wall-normal direction for the low-resolution data.
    :param us:                      Integer indicating the upsampling ratio between the low- and high-resolution data.
    :param generator:               Tensorflow object containing the genertaor architecture.
    :param discriminator:           Tensorflow object containing the discriminator architecture.
    :param generator_optimizer:     Tensorflow object containing the optimizer for the generator architecture.
    :param discriminator_optimizer: Tensorflow object containing the optimizer for the discriminator arquitecture.
    :return:
    """

    """
        Define checkpoint object
    """

    # Define path to checkpoint directory

    checkpoint_dir = f"{root_folder}models/checkpoints_{model_name}"

    # Define checkpoint prefix

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    # Generate checkpoint object to track the generator and discriminator architectures and optimizers

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )

    # Restore the checkpoint with the last stored state

    log = np.genfromtxt(f"{root_folder}logs/log_{model_name}.log", delimiter=',', skip_header=1)


    ckpt = os.listdir(checkpoint_dir)

    saving_freq = int(log.shape[0] / len([string for string in ckpt if re.search(re.compile(f'.index'), string)]))

    for idx in range(2*saving_freq - 1, log.shape[0], saving_freq):

        if log[idx, 3] > log[idx - saving_freq, 3]:

            ckpt_idx = (idx - saving_freq + 1) / saving_freq

            break
    
    ckpt_file = f"{root_folder}models/checkpoints_{model_name}/ckpt-{int(ckpt_idx)}"
    
    checkpoint.restore(ckpt_file).expect_partial()

    """
        Compute predictions
    """

    # Prealocate memory for high- and low-resolution data
    
    lr_target = np.zeros((n_samp, ny, nx, channels), np.float32)
    hr_target = np.zeros((n_samp, ny*us, nx*us, channels), np.float32)
    hr_predic = np.zeros((n_samp, ny*us, nx*us, channels), np.float32)
    dns_target = np.zeros((n_samp, ny*us, nx*us, channels), np.float32)
    cbc_predic = np.zeros((n_samp, ny*us, nx*us, channels), np.float32)
    fl_target = np.zeros((n_samp, ny*us, nx*us, 1), np.float32)

    # Define iterator object

    itr = iter(dataset_test)

    # Iterate over the number of testing samples

    for idx in tqdm(range(n_samp)):

        # Extract sample data
        piv, ptv, dns, cbc, flag, xlr, ylr, xhr, yhr = next(itr) 
        # Save data into global arrays

        lr_target[idx] = piv.numpy()
        hr_target[idx] = ptv.numpy() 
        dns_target[idx] = dns.numpy() 
        cbc_predic[idx] = cbc.numpy() 
        fl_target[idx] = flag.numpy()

        # Compute predictions

        hr_predic[idx] = generator(np.expand_dims(lr_target[idx], axis=0), training=False)
    
    """
        Scale data
    """

    # Define path to file containing scaling value

    filename = f"{root_folder}tfrecords/scaling_us{us}_noise_{noise}.npz"

    if channels == 2:

        # Load mean velocity values in the streamwise and wall-normal directions for low- and high-resolution data

        Upiv_mean = np.expand_dims(np.load(filename)['Upiv_mean'], axis=0)
        Vpiv_mean = np.expand_dims(np.load(filename)['Vpiv_mean'], axis=0)
        Uptv_mean = np.expand_dims(np.load(filename)['Uptv_mean'], axis=0)
        Vptv_mean = np.expand_dims(np.load(filename)['Vptv_mean'], axis=0)

        # Load standard deviation velocity values in the streamwise and wall-normal directions for low- and high-resolution data

        Upiv_std = np.expand_dims(np.load(filename)['Upiv_std'], axis=0)
        Vpiv_std = np.expand_dims(np.load(filename)['Vpiv_std'], axis=0)
        Uptv_std = np.expand_dims(np.load(filename)['Uptv_std'], axis=0)
        Vptv_std = np.expand_dims(np.load(filename)['Vptv_std'], axis=0)
        
        # Scale data

        lr_target[:, :, :, 0] = (lr_target[:, :, :, 0] * Upiv_std + Upiv_mean) * res
        lr_target[:, :, :, 1] = (lr_target[:, :, :, 1] * Vpiv_std + Vpiv_mean) * res
        hr_target[:, :, :, 0] = (hr_target[:, :, :, 0] * Uptv_std + Uptv_mean * fl_target[:, :, :, 0]) * res
        hr_target[:, :, :, 1] = (hr_target[:, :, :, 1] * Vptv_std + Vptv_mean * fl_target[:, :, :, 0]) * res
        hr_predic[:, :, :, 0] = (hr_predic[:, :, :, 0] * Uptv_std + Uptv_mean) * res
        hr_predic[:, :, :, 1] = (hr_predic[:, :, :, 1] * Vptv_std + Vptv_mean) * res
    
    elif channels == 1:

        Tpiv_mean = np.expand_dims(np.load(filename)['Tpiv_mean'], axis=0)
        Tptv_mean = np.expand_dims(np.load(filename)['Tptv_mean'], axis=0)

        Tpiv_std = np.expand_dims(np.load(filename)['Tpiv_std'], axis=0)
        Tptv_std = np.expand_dims(np.load(filename)['Tptv_std'], axis=0)
        
        # Scale data

        lr_target[:, :, :, 0] = (lr_target[:, :, :, 0] * Tpiv_std + Tpiv_mean) * res
        hr_target[:, :, :, 0] = (hr_target[:, :, :, 0] * Tptv_std + Tptv_mean * fl_target[:, :, :, 0]) * res
        hr_predic[:, :, :, 0] = (hr_predic[:, :, :, 0] * Tptv_std + Tptv_mean) * res

    """
        Save predictions
    """

    # Define saving path

    filename = f"{root_folder}results/predictions_{model_name}.npz"

    # Save data

    np.savez(
        filename,
        lr_target=lr_target,
        hr_target=hr_target,
        hr_predic=hr_predic,
        fl_target=fl_target,
        dns_target=dns_target,
        cbc_predic=cbc_predic,
        xlr=xlr,
        ylr=ylr,
        xhr=xhr,
        yhr=yhr
    )

    filename = f"{root_folder}results/predictions_us{us:02d}_{model_name}.mat"

    # Save data

    sio.savemat(
        filename,
        {'lr_target' : lr_target,
        'hr_target' : hr_target,
        'hr_predic' : hr_predic,
        'fl_target' : fl_target,
        'dns_target': dns_target,
        'cbc_predic': cbc_predic,
        'xlr': xlr,
        'ylr': ylr,
        'xhr': xhr,
        'yhr': yhr}
    )

    return




"""
    Old
"""

def compute_metric_mse(dns_target, hr_predic, Udns_std, Vdns_std):

    error_mse_u = np.sqrt(np.mean((dns_target[:, :, :, 0] - hr_predic[:, :, :, 0])**2))
    error_mse_v = np.sqrt(np.mean((dns_target[:, :, :, 1] - hr_predic[:, :, :, 1])**2))
    # error_mse_u = np.nanmean((dns_target[:, :, :, 0] - hr_predic[:, :, :, 0])**2 / Udns_std  / Udns_std, axis=0)
    # error_mse_u[np.isinf(error_mse_u)] = np.nan
    # print((dns_target[0, :, :, 0] - hr_predic[0, :, :, 0]))
    # plt.pcolor((dns_target[0, :, :, 0] - hr_predic[0, :, :, 0])**2 / Udns_std  / Udns_std,vmin=0,vmax=1)
    # plt.savefig('test.png')

    # print(np.nanmean(error_mse_u))
    # lll
    # error_mse_u = np.nanmean((dns_target[:, :, :, 0] - hr_predic[:, :, :, 0])**2 / Udns_std  / Udns_std)
    # error_mse_v = np.nanmean((dns_target[:, :, :, 1] - hr_predic[:, :, :, 1])**2 / Vdns_std  / Vdns_std)

    print("--------------------------------------------")
    print("Mean-squared error")
    print(f"Streamwise velocity: {error_mse_u:0.4f}")
    print(f"Wall-normal velocity: {error_mse_v:0.4f}")
    print("\n")

    return error_mse_u, error_mse_v


def plot_mse(case, us, model_name, dns_target, hr_predic, deptv_predic, Udns_mean, Vdns_mean, yhr):

    error_gan_u = np.mean((dns_target[:, :, :, 0] - hr_predic[:, :, :, 0])**2 / Udns_mean, axis=(0, 2))
    error_gan_v = np.mean((dns_target[:, :, :, 1] - hr_predic[:, :, :, 1])**2 / Vdns_mean, axis=(0, 2))

    error_deptv_u = np.mean((dns_target[:, :, :, 0] - deptv_predic[:, :, :, 0])**2 / Udns_mean, axis=(0, 2))
    error_deptv_v = np.mean((dns_target[:, :, :, 1] - deptv_predic[:, :, :, 1])**2 / Vdns_mean, axis=(0, 2))

    yplus = yhr / 512 * 0.0499 / 0.00005

    figure = TheArtist()
    figure.generate_figure_environment(cols=2, rows=1, fig_width_pt=384, ratio='golden', regular=True)

    figure.plot_lines_semi_x(yplus, error_gan_u, 0, 0, color='k', linewidth=1, linestyle='-', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    figure.plot_lines_semi_x(yplus, error_gan_v, 0, 1, color='k', linewidth=1, linestyle='-', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)

    figure.plot_lines_semi_x(yplus, error_deptv_u, 0, 0, color='r', linewidth=1, linestyle='--', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    figure.plot_lines_semi_x(yplus, error_deptv_v, 0, 1, color='r', linewidth=1, linestyle='--', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    
    figure.set_labels(['$y^+$', '$\epsilon _u$'], 0, 0)
    figure.set_labels(['$y^+$', '$\epsilon _v$'], 0, 1,labelpad=[None, 1])
    figure.set_adjust(wspace=0.3)
    figure.set_axis_lims([[1,1100], [0.001, 1]], 0, 0)
    figure.set_axis_lims([[1,1100], [0, 0.05]], 0, 1)
    figure.axs[0,0].set_yscale('log')

    figure.save_figure(f"./figs/{case}_upsampling-{us:02}_{model_name}_mse", fig_format='pdf', dots_per_inch=1000)



def compute_predictions_sst(root_folder, model_name, dataset_test, n_samp, nx, ny, res, us, generator, discriminator, generator_optimizer, discriminator_optimizer):
    """
        Training logic for GAN architectures.

    :param root_folder:             String containg the folder where data is stored
    :param model_name:              String containing the assigned name to the model, for storage purposes.
    :param dataset_test:            Tensorflow pipeline for the testing dataset.
    :param n_samp:                  Integer indicating the number of samples in the testing dataset
    :param nx:                      Integer indicating the grid points in the streamwise direction for the low-resolution data.
    :param ny:                      Integer indicating the grid points in the wall-normal direction for the low-resolution data.
    :param us:                      Integer indicating the upsampling ratio between the low- and high-resolution data.
    :param generator:               Tensorflow object containing the genertaor architecture.
    :param discriminator:           Tensorflow object containing the discriminator architecture.
    :param generator_optimizer:     Tensorflow object containing the optimizer for the generator architecture.
    :param discriminator_optimizer: Tensorflow object containing the optimizer for the discriminator arquitecture.
    :return:
    """

    """
        Define checkpoint object
    """

    # Define path to checkpoint directory

    checkpoint_dir = f"{root_folder}models/checkpoints_{model_name}"

    # Define checkpoint prefix

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    # Generate checkpoint object to track the generator and discriminator architectures and optimizers

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )

    # Restore the checkpoint with the last stored state

    log = np.genfromtxt(f"{root_folder}logs/log_{model_name}.log", delimiter=',', skip_header=1)


    ckpt = os.listdir(checkpoint_dir)

    saving_freq = int(log.shape[0] / len([string for string in ckpt if re.search(re.compile(f'.index'), string)]))

    for idx in range(2*saving_freq - 1, log.shape[0], saving_freq):

        if log[idx, 3] > log[idx - saving_freq, 3]:

            ckpt_idx = (idx - saving_freq + 1) / saving_freq

            break

        else:

            ckpt_idx = (log.shape[0] - 1) / saving_freq
    
    ckpt_file = f"{root_folder}/models/checkpoints_{model_name}/ckpt-{int(ckpt_idx)}"
    
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    checkpoint.restore(ckpt_file).expect_partial()

    """
        Compute predictions
    """

    # Prealocate memory for high- and low-resolution data
    
    lr_target = np.zeros((n_samp, ny, nx, 1), np.float32)
    hr_target = np.zeros((n_samp, ny*us, nx*us, 1), np.float32)
    hr_predic = np.zeros((n_samp, ny*us, nx*us, 1), np.float32)
    dns_target = np.zeros((n_samp, ny*us, nx*us, 1), np.float32)
    fl_target = np.zeros((n_samp, ny*us, nx*us, 1), np.float32)

    # Define iterator object

    itr = iter(dataset_test)

    # Iterate over the number of testing samples

    for idx in tqdm(range(n_samp)):

        # Extract sample data

        piv, ptv, dns, flag, xlr, ylr, xhr, yhr = next(itr)      

        # Save data into global arrays

        lr_target[idx] = piv.numpy()
        hr_target[idx] = ptv.numpy() 
        dns_target[idx] = dns.numpy() 
        fl_target[idx] = flag.numpy()

        # Compute predictions

        hr_predic[idx] = generator(np.expand_dims(lr_target[idx], axis=0), training=False)
   
    """
        Scale data
    """

    # Define path to file containing scaling value

    filename = f"{root_folder}tfrecords/scaling.npz"

    # Load mean velocity values in the streamwise and wall-normal directions for low- and high-resolution data

    Tpiv_mean = np.expand_dims(np.load(filename)['Tpiv_mean'], axis=0)
    Tptv_mean = np.expand_dims(np.load(filename)['Tptv_mean'], axis=0)

    # Load standard deviation velocity values in the streamwise and wall-normal directions for low- and high-resolution data

    Tpiv_std = np.expand_dims(np.load(filename)['Tpiv_std'], axis=0)
    Tptv_std = np.expand_dims(np.load(filename)['Tptv_std'], axis=0)

    # Scale data

    lr_target[:, :, :, 0] = (lr_target[:, :, :, 0] * Tpiv_std + Tpiv_mean) * res
    hr_target[:, :, :, 0] = (hr_target[:, :, :, 0] * Tptv_std + Tptv_mean * fl_target[:, :, :, 0]) * res
    hr_predic[:, :, :, 0] = (hr_predic[:, :, :, 0] * Tptv_std + Tptv_mean) * res

    """
        Save predictions
    """

    # Define saving path

    filename = f"{root_folder}results/predictions_{model_name}.npz"

    # Save data

    np.savez(
        filename,
        lr_target=lr_target,
        hr_target=hr_target,
        hr_predic=hr_predic,
        fl_target=fl_target,
        dns_target=dns_target,
        xlr=xlr,
        ylr=ylr,
        xhr=xhr,
        yhr=yhr
    )

    filename = f"{root_folder}results/predictions_us{us:02d}_{model_name}.mat"

    # Save data

    sio.savemat(
        filename,
        {'lr_target' : lr_target,
        'hr_target' : hr_target,
        'hr_predic' : hr_predic,
        'fl_target' : fl_target,
        'dns_target': dns_target,
        'xlr': xlr,
        'ylr': ylr,
        'xhr': xhr,
        'yhr': yhr}
    )

    return


def compute_spectra_taylor_hypothesis(root_folder, time_folder, model_name, dataset_time, n_samp, nx, ny, us, generator, discriminator, generator_optimizer, discriminator_optimizer):

    """
        Define checkpoint object
    """

    # Define path to checkpoint directory

    checkpoint_dir = f"{root_folder}models/checkpoints_{model_name}"

    # Define checkpoint prefix

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    # Generate checkpoint object to track the generator and discriminator architectures and optimizers

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )

    # Restore the checkpoint with the last stored state

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    """
        Compute predictions
    """

    # Prealocate memory for high- and low-resolution data

    lr_target = np.zeros((n_samp, ny, nx, 2), np.float32)
    hr_predic = np.zeros((n_samp, ny*us, nx*us, 2), np.float32)
    dns_target = np.zeros((n_samp, ny*us, nx*us, 2), np.float32)

    # Define iterator object

    itr = iter(dataset_time)

    # Iterate over the number of testing samples

    for idx in tqdm(range(n_samp)):

        # Extract sample data

        piv, _, dns, _ = next(itr)

        # Save data into global arrays

        lr_target[idx] = piv.numpy()
        dns_target[idx] = dns.numpy() 

        # Compute predictions

        hr_predic[idx] = generator(np.expand_dims(lr_target[idx], axis=0), training=False)

    """
        Scale data
    """

    # Define path to file containing scaling value

    filename = f"{root_folder}tfrecords/scaling.npz"

    # Load mean velocity values in the streamwise and wall-normal directions for low- and high-resolution data

    Upiv_mean = np.expand_dims(np.load(filename)['Upiv_mean'], axis=0)
    Vpiv_mean = np.expand_dims(np.load(filename)['Vpiv_mean'], axis=0)
    Uptv_mean = np.expand_dims(np.load(filename)['Uptv_mean'], axis=0)
    Vptv_mean = np.expand_dims(np.load(filename)['Vptv_mean'], axis=0)
    Udns_mean = np.expand_dims(np.load(filename)['Udns_mean'], axis=0)
    Vdns_mean = np.expand_dims(np.load(filename)['Vdns_mean'], axis=0)

    # Load standard deviation velocity values in the streamwise and wall-normal directions for low- and high-resolution data

    Upiv_std = np.expand_dims(np.load(filename)['Upiv_std'], axis=0)
    Vpiv_std = np.expand_dims(np.load(filename)['Vpiv_std'], axis=0)
    Uptv_std = np.expand_dims(np.load(filename)['Uptv_std'], axis=0)
    Vptv_std = np.expand_dims(np.load(filename)['Vptv_std'], axis=0)
    Udns_std = np.expand_dims(np.load(filename)['Udns_std'], axis=0)
    Vdns_std = np.expand_dims(np.load(filename)['Vdns_std'], axis=0)

    # Scale data

    lr_target[:, :, :, 0] = lr_target[:, :, :, 0] * Upiv_std + Upiv_mean
    lr_target[:, :, :, 1] = lr_target[:, :, :, 1] * Vpiv_std + Vpiv_mean
    hr_predic[:, :, :, 0] = hr_predic[:, :, :, 0] * Uptv_std + Uptv_mean
    hr_predic[:, :, :, 1] = hr_predic[:, :, :, 1] * Vptv_std + Vptv_mean

    """
        Compute spectra
    """

    kx = np.arange(1, nx * us / 2 + 0.1, 1)
    lx = 2 / kx * 1000
    kx = np.expand_dims(kx, axis=1)

    print(kx.shape)
    print(Udns_mean.shape)

    spectra_uu_dns = kx * np.mean(np.abs(np.fft.fft(dns_target[:, :, :, 0] - Udns_mean, axis=0))**2, axis=2)[:int(nx * us / 2), :]
    print(spectra_uu_dns.shape)

    """
        Define colormaps
    """

    # DNS

    cmap_dns = matplotlib.cm.get_cmap("Greys").copy()

    """
        TheArtist environment
    """

    grid = sio.loadmat(f"{root_folder}Grid.mat")
    yplus = grid['Y'][:, 0] / 500 * 0.0499 / 0.00005

    # Initialize class

    figure = TheArtist()

    # Initialize figure

    print(spectra_uu_dns)

    figure.generate_figure_environment(cols=1, rows=1, fig_width_pt=384, ratio=0.6, regular=True)

    figure.plot_panel_contourf(yplus, lx, spectra_uu_dns/ spectra_uu_dns.max(), 0, 0, cmap=cmap_dns, clims=[0, 1], levels=[0.1, 0.3, 0.5, 0.7, 0.9], extend='both', norm=matplotlib.colors.Normalize(vmin=0, vmax=1))

    figure.set_labels(['$y^+$', '$\lambda^+$'], 0, 0, labelpad=[None, None], flip = [False, False])

    figure.axs[0,0].set_xscale('log')
    figure.axs[0,0].set_yscale('log')

    figure.save_figure(f"./figs/channel_upsampling-{us:02}_{model_name}_spectra", fig_format='pdf', dots_per_inch=1000)

    return


def plot_averaged_error(case, us, model_name, lr_target, hr_predic, dns_target, fl_target):

    error_hr_predic_rmse_u = np.std(dns_target[:, :, :, 0] - hr_predic[:, :, :, 0], axis=0)
    error_hr_predic_rmse_v = np.std(dns_target[:, :, :, 1] - hr_predic[:, :, :, 1], axis=0)

    error_lr_target_rmse_u = np.std(dns_target[:, ::us, ::us, 0] - lr_target[:, :, :, 0], axis=0)
    error_lr_target_rmse_v = np.std(dns_target[:, ::us, ::us, 1] - lr_target[:, :, :, 1], axis=0)

    """
        Define colormaps
    """

    # Error

    cmap = matplotlib.cm.get_cmap("hot_r").copy()

    # Assign black to NaN values

    cmap.set_bad(color='k')

    """
        TheArtist environment
    """

    # Initialize class

    figure = TheArtist()

    # Initialize figure

    figure.generate_figure_environment(cols=2, rows=2, fig_width_pt=384, ratio=0.6, regular=True)

    figure.plot_panel_imshow(error_hr_predic_rmse_u, 0, 0, vmin=0, vmax=0.1, origin='upper', extent=[0, 2, 0, 1], cmap=cmap)
    figure.plot_panel_imshow(error_lr_target_rmse_u, 0, 1, vmin=0, vmax=0.1, origin='upper', extent=[0, 2, 0, 1], cmap=cmap)

    figure.plot_panel_imshow(error_hr_predic_rmse_v, 1, 0, vmin=0, vmax=0.1, origin='upper', extent=[0, 2, 0, 1], cmap=cmap)
    figure.plot_panel_imshow(error_lr_target_rmse_v, 1, 1, vmin=0, vmax=0.1, origin='upper', extent=[0, 2, 0, 1], cmap=cmap)

    figure.set_labels([None, '$y/h$'], 0, 0, labelpad=[None, None], flip = [False, False])
    figure.set_labels(['$x/h$', '$y/h$'], 1, 0, labelpad=[None, None], flip = [False, False])
    figure.set_labels(['$x/h$', None], 1, 1, labelpad=[None, None], flip = [False, False])

    figure.set_ticks([[], [0, 1]], 0, 0)
    figure.set_ticks([[0, 1, 2], [0, 1]], 1, 0)
    figure.set_ticks([[], []], 0, 1)
    figure.set_ticks([[0, 1, 2], []], 1, 1)

    figure.set_tick_params(0, 0, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    figure.set_tick_params(1, 0, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    figure.set_tick_params(0, 1, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    figure.set_tick_params(1, 1, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)

    figure.set_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.01)

    cax = figure.fig.add_axes([0.3, 0.9, 0.4, 0.03])

    figure.fig.colorbar(figure.im[0], cax=cax, orientation='horizontal', shrink=0.7, extendfrac=0, ticks=[0, 0.1])

    figure.save_figure(f"./figs/{case}_upsampling-{us:02}_{model_name}_averaged-error", fig_format='pdf', dots_per_inch=1000)

    return


def plot_instantaneous_error(case, us, model_name, lr_target, hr_predic, dns_target, fl_target, field=0):

    """
        Define colormaps
    """

    # Error

    cmap = matplotlib.cm.get_cmap("hot_r").copy()

    # Assign black to NaN values

    cmap.set_bad(color='k')

    """
        TheArtist environment
    """

    # Initialize class

    figure = TheArtist()

    # Initialize figure

    figure.generate_figure_environment(cols=2, rows=2, fig_width_pt=384, ratio=0.6, regular=True)

    figure.plot_panel_imshow(((hr_predic[0, :, :, 0]-dns_target[0, :, :, 0])**2)**0.5,       0, 0, vmin=0, vmax=0.2, origin='upper', extent=[0, 2, 0, 1], cmap=cmap)
    figure.plot_panel_imshow(((lr_target[0, :, :, 0]-dns_target[0, ::us, ::us, 0])**2)**0.5, 0, 1, vmin=0, vmax=0.2, origin='upper', extent=[0, 2, 0, 1], cmap=cmap)

    figure.plot_panel_imshow(((hr_predic[0, :, :, 1]-dns_target[0, :, :, 1])**2)**0.5,       1, 0, vmin=0, vmax=0.2, origin='upper', extent=[0, 2, 0, 1], cmap=cmap)
    figure.plot_panel_imshow(((lr_target[0, :, :, 1]-dns_target[0, ::us, ::us, 1])**2)**0.5, 1, 1, vmin=0, vmax=0.2, origin='upper', extent=[0, 2, 0, 1], cmap=cmap)

    figure.set_labels([None, '$y/h$'], 0, 0, labelpad=[None, None], flip = [False, False])
    figure.set_labels(['$x/h$', '$y/h$'], 1, 0, labelpad=[None, None], flip = [False, False])
    figure.set_labels(['$x/h$', None], 1, 1, labelpad=[None, None], flip = [False, False])

    figure.set_ticks([[], [0, 1]], 0, 0)
    figure.set_ticks([[0, 1, 2], [0, 1]], 1, 0)
    figure.set_ticks([[], []], 0, 1)
    figure.set_ticks([[0, 1, 2], []], 1, 1)

    figure.set_tick_params(0, 0, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    figure.set_tick_params(1, 0, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    figure.set_tick_params(0, 1, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    figure.set_tick_params(1, 1, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)

    figure.set_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.01)

    figure.save_figure(f"./figs/{case}_upsampling-{us:02}_{model_name}_instantaneous-error", fig_format='pdf', dots_per_inch=1000)

    return


def plot_instantaneous_field(case, us, model_name, lr_target, hr_target, hr_predic, dns_target, fl_target, xhr, yhr, field=0):

    if case == 'channel':
        hr_target[np.repeat(fl_target, 2, axis=3) == 0] = np.nan
        """
            TheArtist environment
        """

        # Initialize class

        figure = TheArtist()

        # Initialize figure

        figure.generate_figure_environment(cols=4, rows=2, fig_width_pt=472, ratio=0.6, regular=True)

        """
            Define colormaps
        """

        # Streamwise velocity

        cmap_u = matplotlib.cm.get_cmap("viridis").copy()

        # Wall-normal velocity
        
        cmap_v = matplotlib.cm.get_cmap("RdYlBu_r").copy()

        # Assign black to NaN values

        cmap_u.set_bad(color='k')
        cmap_v.set_bad(color='k')

        figure.plot_panel_imshow(lr_target[0, :, :, 0], 0, 0, vmin=0.4, vmax=1.2, origin='upper', extent=[0, 2, 0, 1], cmap=cmap_u)
        figure.plot_panel_imshow(hr_target[0, :, :, 0], 0, 1, vmin=0.4, vmax=1.2, origin='upper', extent=[0, 2, 0, 1], cmap=cmap_u)
        figure.plot_panel_imshow(hr_predic[0, :, :, 0], 0, 2, vmin=0.4, vmax=1.2, origin='upper', extent=[0, 2, 0, 1], cmap=cmap_u)
        figure.plot_panel_imshow(dns_target[0, :, :, 0], 0, 3, vmin=0.4, vmax=1.2, origin='upper', extent=[0, 2, 0, 1], cmap=cmap_u)
        
        figure.plot_panel_imshow(lr_target[0, :, :, 1], 1, 0, vmin=-0.1, vmax=0.1, origin='upper', extent=[0, 2, 0, 1], cmap=cmap_v)
        figure.plot_panel_imshow(hr_target[0, :, :, 1], 1, 1, vmin=-0.1, vmax=0.1, origin='upper', extent=[0, 2, 0, 1], cmap=cmap_v)
        figure.plot_panel_imshow(hr_predic[0, :, :, 1], 1, 2, vmin=-0.1, vmax=0.1, origin='upper', extent=[0, 2, 0, 1], cmap=cmap_v)
        figure.plot_panel_imshow(dns_target[0, :, :, 1], 1, 3, vmin=-0.1, vmax=0.1, origin='upper', extent=[0, 2, 0, 1], cmap=cmap_v)

        figure.set_labels([None, '$y/h$'], 0, 0, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['$x/h$', '$y/h$'], 1, 0, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['$x/h$', None], 1, 1, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['$x/h$', None], 1, 2, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['$x/h$', None], 1, 3, labelpad=[None, None], flip = [False, False])

        figure.set_ticks([[], [0, 1]], 0, 0)
        figure.set_ticks([[0, 1, 2], [0, 1]], 1, 0)
        figure.set_ticks([[], []], 0, 1)
        figure.set_ticks([[0, 1, 2], []], 1, 1)
        figure.set_ticks([[], []], 0, 2)
        figure.set_ticks([[0, 1, 2], []], 1, 2)
        figure.set_ticks([[], []], 0, 3)
        figure.set_ticks([[0, 1, 2], []], 1, 3)

        figure.set_tick_params(0, 0, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(1, 0, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(0, 1, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(1, 1, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(0, 2, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(1, 2, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(0, 3, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(1, 3, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)

        caxu = figure.fig.add_axes([0.92, 0.55, 0.015, 0.28])
        caxv = figure.fig.add_axes([0.92, 0.16, 0.015, 0.28])

        figure.fig.colorbar(figure.im[0], cax=caxu, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[0.4, 0.8, 1.2])
        figure.fig.colorbar(figure.im[4], cax=caxv, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-0.1, 0, 0.1])


    elif case == 'pinball':
        hr_target[np.repeat(fl_target, 2, axis=3) == 0] = np.nan
        """
            TheArtist environment
        """

        # Initialize class

        figure = TheArtist()

        # Initialize figure

        figure.generate_figure_environment(cols=4, rows=2, fig_width_pt=510, ratio=0.5, regular=True)

        Res = 25
        xmin = -5
        ymin = -4 + 4 / Res
        xhr = xhr / Res + xmin
        yhr = yhr / Res + ymin

        """
            Define colormaps
        """

        # Streamwise velocity

        cmap_u = matplotlib.cm.get_cmap("viridis").copy()

        # Wall-normal velocity
        
        cmap_v = matplotlib.cm.get_cmap("RdYlBu_r").copy()

        # Assign black to NaN values

        cmap_u.set_bad(color='k')
        cmap_v.set_bad(color='k')

        figure.plot_panel_imshow(lr_target[0, :, :, 0], 0, 0, vmin=-1.5, vmax=1.5, origin='upper',  extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_u)
        figure.plot_panel_imshow(hr_target[0, :, :, 0], 0, 1, vmin=-1.5, vmax=1.5, origin='upper',  extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_u)
        figure.plot_panel_imshow(hr_predic[0, :, :, 0], 0, 2, vmin=-1.5, vmax=1.5, origin='upper',  extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_u)
        figure.plot_panel_imshow(dns_target[0, :, :, 0], 0, 3, vmin=-1.5, vmax=1.5, origin='upper', extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_u)
        
        figure.plot_panel_imshow(lr_target[0, :, :, 1], 1, 0, vmin=-0.7, vmax=0.7, origin='upper',  extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_v)
        figure.plot_panel_imshow(hr_target[0, :, :, 1], 1, 1, vmin=-0.7, vmax=0.7, origin='upper',  extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_v)
        figure.plot_panel_imshow(hr_predic[0, :, :, 1], 1, 2, vmin=-0.7, vmax=0.7, origin='upper',  extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_v)
        figure.plot_panel_imshow(dns_target[0, :, :, 1], 1, 3, vmin=-0.7, vmax=0.7, origin='upper', extent=[xhr.min(), xhr.max(), yhr.min(), yhr.max()], cmap=cmap_v)

        figure.set_labels([None, '$y/D$'], 0, 0, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['$x/D$', '$y/D$'], 1, 0, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['$x/D$', None], 1, 1, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['$x/D$', None], 1, 2, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['$x/D$', None], 1, 3, labelpad=[None, None], flip = [False, False])

        for idx in range(4):
            for idy in range(2):

                draw_circle1 = plt.Circle((- 1.299, 0), 0.5, edgecolor=None, facecolor='gray')
                draw_circle2 = plt.Circle((0, 0.75), 0.5,    edgecolor=None, facecolor='gray')
                draw_circle3 = plt.Circle((0, -0.75), 0.5,   edgecolor=None, facecolor='gray')
                figure.axs[idy, idx].add_artist(draw_circle1)
                figure.axs[idy, idx].add_artist(draw_circle2)
                figure.axs[idy, idx].add_artist(draw_circle3)

        figure.set_ticks([[], [-4, 0, 4]], 0, 0)
        figure.set_ticks([[-5, 0, 5, 10, 15], [-4, 0, 4]], 1, 0)
        figure.set_ticks([[], []], 0, 1)
        figure.set_ticks([[-5, 0, 5, 10, 15], []], 1, 1)
        figure.set_ticks([[], []], 0, 2)
        figure.set_ticks([[-5, 0, 5, 10, 15], []], 1, 2)
        figure.set_ticks([[], []], 0, 3)
        figure.set_ticks([[-5, 0, 5, 10, 15], []], 1, 3)

        figure.set_tick_params(0, 0, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(1, 0, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(0, 1, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(1, 1, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(0, 2, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(1, 2, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(0, 3, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(1, 3, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)


        caxu = figure.fig.add_axes([0.92, 0.57, 0.015, 0.24])
        caxv = figure.fig.add_axes([0.92, 0.18, 0.015, 0.24])

        figure.fig.colorbar(figure.im[0], cax=caxu, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-1.5, 0, 1.5])
        figure.fig.colorbar(figure.im[4], cax=caxv, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-0.7, 0, 0.7])


    elif case == 'exptbl':
        hr_target[np.repeat(fl_target, 2, axis=3) == 0] = np.nan
        """
            TheArtist environment
        """

        # Initialize class

        figure = TheArtist()

        # Initialize figure

        figure.generate_figure_environment(cols=3, rows=2, fig_width_pt=472, ratio=1.0, regular=True)

        """
            Define colormaps
        """

        # Streamwise velocity

        cmap_u = matplotlib.cm.get_cmap("viridis").copy()

        # Wall-normal velocity
        
        cmap_v = matplotlib.cm.get_cmap("RdYlBu_r").copy()

        # Assign black to NaN values

        cmap_u.set_bad(color='k')
        cmap_v.set_bad(color='k')
        
        figure.plot_panel_imshow(lr_target[0, :, :, 0], 0, 0, vmin=0.4, vmax=1.2, origin='upper', extent=[0, 1, 0, 1], cmap=cmap_u)
        figure.plot_panel_imshow(hr_target[0, :, :, 0], 0, 1, vmin=0.4, vmax=1.2, origin='upper', extent=[0, 1, 0, 1], cmap=cmap_u)
        figure.plot_panel_imshow(hr_predic[0, :, :, 0], 0, 2, vmin=0.4, vmax=1.2, origin='upper', extent=[0, 1, 0, 1], cmap=cmap_u)

        figure.plot_panel_imshow(lr_target[0, :, :, 1], 1, 0, vmin=-0.1, vmax=0.1, origin='upper', extent=[0, 1, 0, 1], cmap=cmap_v)
        figure.plot_panel_imshow(hr_target[0, :, :, 1], 1, 1, vmin=-0.1, vmax=0.1, origin='upper', extent=[0, 1, 0, 1], cmap=cmap_v)
        figure.plot_panel_imshow(hr_predic[0, :, :, 1], 1, 2, vmin=-0.1, vmax=0.1, origin='upper', extent=[0, 1, 0, 1], cmap=cmap_v)
    
        figure.set_labels([None, '$y/h$'], 0, 0, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['$x/h$', '$y/h$'], 1, 0, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['$x/h$', None], 1, 1, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['$x/h$', None], 1, 2, labelpad=[None, None], flip = [False, False])
     
        figure.set_ticks([[], [0, 1]], 0, 0)
        figure.set_ticks([[0, 1], [0, 1]], 1, 0)
        figure.set_ticks([[], []], 0, 1)
        figure.set_ticks([[0, 1], []], 1, 1)
        figure.set_ticks([[], []], 0, 2)
        figure.set_ticks([[0, 1], []], 1, 2)

        figure.set_tick_params(0, 0, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(1, 0, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(0, 1, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(1, 1, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(0, 2, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(1, 2, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)

        caxu = figure.fig.add_axes([0.92, 0.55, 0.015, 0.28])
        caxv = figure.fig.add_axes([0.92, 0.16, 0.015, 0.28])

        figure.fig.colorbar(figure.im[0], cax=caxu, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[0.4, 0.8, 1.2])
        figure.fig.colorbar(figure.im[4], cax=caxv, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-0.1, 0, 0.1])
    
    
    elif case == 'sst':
        hr_target[fl_target == 0] = np.nan
        """
            TheArtist environment
        """

        # Initialize class

        figure = TheArtist()

        # Initialize figure

        figure.generate_figure_environment(cols=4, rows=1, fig_width_pt=472, ratio=0.6, regular=True)

        """
            Define colormaps
        """

        # Streamwise velocity

        cmap_t = matplotlib.cm.get_cmap("plasma").copy()


        # Assign black to NaN values

        cmap_t.set_bad(color='w')

        figure.plot_panel_imshow(lr_target[0, :, :, 0], 0, 0, vmin=0, vmax=30, origin='lower', extent=[0, 360, -90, 90], cmap=cmap_t)
        figure.plot_panel_imshow(hr_target[0, :, :, 0], 0, 1, vmin=0, vmax=30, origin='lower', extent=[0, 360, -90, 90], cmap=cmap_t)
        figure.plot_panel_imshow(hr_predic[0, :, :, 0], 0, 2, vmin=0, vmax=30, origin='lower', extent=[0, 360, -90, 90], cmap=cmap_t)
        figure.plot_panel_imshow(dns_target[0, :, :, 0], 0, 3, vmin=0, vmax=30, origin='lower', extent=[0, 360, -90, 90], cmap=cmap_t)
       
        figure.set_labels(['Longitude', 'Latitude'], 0, 0, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['Longitude', None], 0, 1, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['Longitude', None], 0, 2, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['Longitude', None], 0, 3, labelpad=[None, None], flip = [False, False])

        figure.set_ticks([[0, 180, 360], [-90, 0, 90]], 0, 0)
        figure.set_ticks([[0, 180, 360], []], 0, 1)
        figure.set_ticks([[0, 180, 360], []], 0, 2)
        figure.set_ticks([[0, 180, 360], []], 0, 3)

        figure.set_tick_params(0, 0, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(0, 1, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(0, 2, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(0, 3, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        
        caxt = figure.fig.add_axes([0.92, 0.1, 0.015, 0.8])

        figure.fig.colorbar(figure.im[0], cax=caxt, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[0, 15, 30])
    

    elif case == 'buoyancy':
        hr_target[fl_target == 0] = np.nan
        """
            TheArtist environment
        """
        
        # Initialize class

        figure = TheArtist()

        # Initialize figure

        figure.generate_figure_environment(cols=4, rows=1, fig_width_pt=472, ratio=1.5, regular=True)

        """
            Define colormaps
        """

        # Streamwise velocity

        cmap_t = matplotlib.cm.get_cmap("seismic").copy()


        # Assign black to NaN values

        cmap_t.set_bad(color='w')

        figure.plot_panel_imshow(lr_target[990, :, :, 0], 0, 0, vmin=-3, vmax=3, origin='lower', extent=[0, 2, 0, 2], cmap=cmap_t)
        figure.plot_panel_imshow(hr_target[990, :, :, 0], 0, 1, vmin=-3, vmax=3, origin='lower', extent=[0, 2, 0, 2], cmap=cmap_t)
        figure.plot_panel_imshow(hr_predic[990, :, :, 0], 0, 2, vmin=-3, vmax=3, origin='lower', extent=[0, 2, 0, 2], cmap=cmap_t)
        figure.plot_panel_imshow(dns_target[990, :, :, 0], 0, 3, vmin=-3, vmax=3, origin='lower', extent=[0, 2, 0, 2], cmap=cmap_t)
       
        figure.set_labels(['$x/L\\pi$', '$y/L\\pi$'], 0, 0, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['$x/L\\pi$', None], 0, 1, labelpad=[None, None], flip = [False, False])
        figure.set_labels(['$x/L\\pi$', None], 0, 2, labelpad=[None, None], flip = [False, False])
        # figure.set_labels(['Longitude', None], 0, 3, labelpad=[None, None], flip = [False, False])

        figure.set_ticks([[0, 2], [0, 2]], 0, 0)
        figure.set_ticks([[0, 2], []], 0, 1)
        figure.set_ticks([[0, 2], []], 0, 2)
        # figure.set_ticks([[0, 180, 360], []], 0, 3)

        figure.set_tick_params(0, 0, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(0, 1, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        figure.set_tick_params(0, 2, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        # figure.set_tick_params(0, 3, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
        
        caxt = figure.fig.add_axes([0.92, 0.2, 0.015, 0.6])

        figure.fig.colorbar(figure.im[0], cax=caxt, orientation='vertical', shrink=1.0, extendfrac=0, ticks=[-3, 0, 3])
    

    figure.set_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.01)

    figure.save_figure(f"./figs/{case}_upsampling-{us:02}_{model_name}_instantaneous-field", fig_format='pdf', dots_per_inch=1000)

    return


def plot_instantaneous_field_stefano(case, us, model_name, lr_target, hr_target, hr_predic, fl_target, field=0):

    """
        Define colormaps
    """

    # Streamwise velocity

    cmap_u = matplotlib.cm.get_cmap("magma").copy()

    # Wall-normal velocity
    
    cmap_v = matplotlib.cm.get_cmap("RdBu_r").copy()

    # Assign black to NaN values

    cmap_u.set_bad(color='k')
    cmap_v.set_bad(color='k')

    """
        TheArtist environment
    """

    # Initialize class

    figure = TheArtist()

    # Initialize figure

    figure.generate_figure_environment(cols=3, rows=2, fig_width_pt=384, ratio=0.6, regular=True)

    hr_target[np.repeat(fl_target, 2, axis=3) == 0] = np.nan

    figure.plot_panel_imshow(lr_target[0, :, :, 0], 0, 0, vmin=0.4, vmax=1.2, origin='upper', extent=[0, 2, 0, 1], cmap=cmap_u)
    figure.plot_panel_imshow(hr_target[0, :, :, 0], 0, 1, vmin=0.4, vmax=1.2, origin='upper', extent=[0, 2, 0, 1], cmap=cmap_u)
    figure.plot_panel_imshow(hr_predic[0, :, :, 0], 0, 2, vmin=0.4, vmax=1.2, origin='upper', extent=[0, 2, 0, 1], cmap=cmap_u)
    
    figure.plot_panel_imshow(lr_target[0, :, :, 1], 1, 0, vmin=-0.1, vmax=0.1, origin='upper', extent=[0, 2, 0, 1], cmap=cmap_v)
    figure.plot_panel_imshow(hr_target[0, :, :, 1], 1, 1, vmin=-0.1, vmax=0.1, origin='upper', extent=[0, 2, 0, 1], cmap=cmap_v)
    figure.plot_panel_imshow(hr_predic[0, :, :, 1], 1, 2, vmin=-0.1, vmax=0.1, origin='upper', extent=[0, 2, 0, 1], cmap=cmap_v)

    figure.set_labels([None, '$y/h$'], 0, 0, labelpad=[None, None], flip = [False, False])
    figure.set_labels(['$x/h$', '$y/h$'], 1, 0, labelpad=[None, None], flip = [False, False])
    figure.set_labels(['$x/h$', None], 1, 1, labelpad=[None, None], flip = [False, False])
    figure.set_labels(['$x/h$', None], 1, 2, labelpad=[None, None], flip = [False, False])

    figure.set_ticks([[], [0, 1]], 0, 0)
    figure.set_ticks([[0, 1, 2], [0, 1]], 1, 0)
    figure.set_ticks([[], []], 0, 1)
    figure.set_ticks([[0, 1, 2], []], 1, 1)
    figure.set_ticks([[], []], 0, 2)
    figure.set_ticks([[0, 1, 2], []], 1, 2)

    figure.set_tick_params(0, 0, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    figure.set_tick_params(1, 0, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    figure.set_tick_params(0, 1, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    figure.set_tick_params(1, 1, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    figure.set_tick_params(0, 2, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)
    figure.set_tick_params(1, 2, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=0)

    figure.set_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.01)

    figure.save_figure(f"./figs/{case}_upsampling-{us:02}_{model_name}_instantaneous-field", fig_format='pdf', dots_per_inch=1000)

    return


def plot_turbulent_statistics(root_folder, case, us, model_name, Upiv_mean, Upiv_std, Uptv_mean, Uptv_std, hr_predic, dns_target, ylr, yhr):

    import matplotlib

    cmap = matplotlib.cm.get_cmap('viridis')

    cpiv = cmap(1.0)
    cptv = cmap(0.6)
    cgan = cmap(0.3)
    cdns = cmap(0.0)

    yplus = yhr / 512 * 0.0499 / 0.00005
    yppiv = ylr / 512 * 0.0499 / 0.00005

    mean_u_dns = np.mean(dns_target[:, ::-1, :, 0], axis=(0, 2)) / 0.0499
    mean_u_ptv = np.mean(Uptv_mean,axis=(0, 2))[::-1] / 0.0499
    mean_u_gan = np.mean(hr_predic[:, ::-1, :, 0], axis=(0, 2)) / 0.0499
    mean_u_piv = np.mean(Upiv_mean,axis=(0, 2))[::-1] / 0.0499

    mean_uu_dns = np.std(dns_target[:, ::-1, :, 0], axis=(0, 2)) / 0.0499
    mean_uu_ptv = np.mean(Uptv_std,axis=(0, 2))[::-1] / 0.0499
    mean_uu_piv = np.mean(Upiv_std,axis=(0, 2))[::-1] / 0.0499
    mean_uu_gan = np.std(hr_predic[:, ::-1, :, 0], axis=(0, 2)) / 0.0499

    figure = TheArtist()
    figure.generate_figure_environment(cols=2, rows=1, fig_width_pt=384, ratio='golden', regular=True)

    figure.plot_lines_semi_x(yplus, mean_u_dns, 0, 0, color=cdns, linewidth=1, linestyle='-', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    figure.plot_lines_semi_x(yplus, mean_u_ptv, 0, 0, color=cptv, linewidth=1, linestyle='--', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    figure.plot_lines_semi_x(yplus, mean_u_gan, 0, 0, color=cpiv, linewidth=1, linestyle=':', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    figure.plot_lines_semi_x(yppiv, mean_u_piv, 0, 0, color=cgan, linewidth=1, linestyle='-.', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)

    figure.plot_lines_semi_x(yplus, mean_uu_dns, 0, 1, color=cdns, linewidth=1, linestyle='-', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    figure.plot_lines_semi_x(yplus, mean_uu_ptv, 0, 1, color=cptv, linewidth=1, linestyle='--', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    figure.plot_lines_semi_x(yplus, mean_uu_gan, 0, 1, color=cpiv, linewidth=1, linestyle=':', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)
    figure.plot_lines_semi_x(yppiv, mean_uu_piv, 0, 1, color=cgan, linewidth=1, linestyle='-.', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2)

    figure.set_labels(['$y^+$', '$U^+$'], 0, 0)
    figure.set_labels(['$y^+$', '$u^+$'], 0, 1,labelpad=[None, 1])
    # figure.set_adjust(wspace=0.1)
    figure.set_axis_lims([[1,1100], [0, 25]], 0, 0)
    figure.set_axis_lims([[1,1100], [0, 4]], 0, 1)

    figure.save_figure(f"./figs/{case}_upsampling-{us:02}_{model_name}_statistics", fig_format='pdf', dots_per_inch=1000)

    return


def plot_nsamples_analysis(n_samples_pinball, error_mse_u_pinball, error_mse_v_pinball, n_samples_channel, error_mse_u_channel, error_mse_v_channel):

    figure = TheArtist(fontsize=6)
    figure.generate_figure_environment(cols=2, rows=1, fig_width_pt=510, ratio='golden', regular=True)

    figure.plot_lines(n_samples_pinball, error_mse_u_pinball, 0, 0, color='r', linewidth=1, linestyle='-', marker='s', markeredgecolor=None, markerfacecolor='r', markersize=2)
    figure.plot_lines(n_samples_pinball, error_mse_v_pinball, 0, 0, color='b', linewidth=1, linestyle='--', marker='o', markeredgecolor=None, markerfacecolor='b', markersize=2)

    figure.plot_lines(n_samples_channel, error_mse_u_channel, 0, 1, color='r', linewidth=1, linestyle='-', marker='s', markeredgecolor=None, markerfacecolor='r', markersize=2)
    figure.plot_lines(n_samples_channel, error_mse_v_channel, 0, 1, color='b', linewidth=1, linestyle='--', marker='o', markeredgecolor=None, markerfacecolor='b', markersize=2)
    
    figure.set_labels(['$n_t$', '$\\varepsilon$'], 0, 0)
    figure.set_labels(['$n_t$', None], 0, 1,labelpad=[None, 1])
    # figure.set_adjust(wspace=0.1)
    figure.set_axis_lims([[0, 4000], [0.25, 0.35]], 0, 0)
    figure.set_axis_lims([[0, 10000], [0.30, 0.50]], 0, 1)

    figure.set_title("Fluidic Pinball", 0, 0)
    figure.set_title("Turbulent Channel Flow", 0, 1)

    blue_line = matplotlib.lines.Line2D([], [], color='r', marker='s', markersize=3, label='$u$')
    red_line = matplotlib.lines.Line2D([], [], color='b', marker='o', markersize=3, label='$v$')
    figure.axs[0,0].legend(handles=[blue_line, red_line], frameon=False)

    blue_line = matplotlib.lines.Line2D([], [], color='r', linestyle='-', marker='s', markersize=3, label='$u$')
    red_line = matplotlib.lines.Line2D([], [], color='b', linestyle='--', marker='o', markersize=3, label='$v$')
    figure.axs[0,1].legend(handles=[blue_line, red_line], frameon=False)
    # figure.set_ticks([None,[0.06, 0.08,0.10,0.12]],0,0)
    # figure.set_ticks([None,[0.1,0.15,0.2,0.25]],0,1)

    figure.save_figure(f"./figs/number-of-training-samples-analysis", fig_format='pdf', dots_per_inch=1000)

    return