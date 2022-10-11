# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:04:07 2021
@author: guemesturb
"""


import os
import re
<<<<<<< HEAD
=======
import sys
# sys.path.insert(1, '/home/aguemes/tools/TheArtist')
>>>>>>> d697111dba0b1b8df36000c0c120a92bf7e61e0e
import matplotlib
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tqdm import tqdm
<<<<<<< HEAD
from TheArtist import TheArtist
=======
import TheArtist
>>>>>>> d697111dba0b1b8df36000c0c120a92bf7e61e0e
import matplotlib.pyplot as plt


def compute_predictions(root_folder, model_name, dataset_test, n_samp, nx, ny, res, us, channels, noise, generator, discriminator, generator_optimizer, discriminator_optimizer, subversion, save_mat=False):
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
    print(ckpt_file)
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
    gap_predic = np.zeros((n_samp, ny*us, nx*us, channels), np.float32)
    fl_target = np.zeros((n_samp, ny*us, nx*us, 1), np.float32)

    # Define iterator object

    itr = iter(dataset_test)

    # Iterate over the number of testing samples

    for idx in tqdm(range(n_samp)):

        # Extract sample data
        piv, ptv, dns, cbc, gap, flag, xlr, ylr, xhr, yhr = next(itr) 
        # Save data into global arrays
        
        lr_target[idx] = piv.numpy()
        hr_target[idx] = ptv.numpy() 
        dns_target[idx] = dns.numpy() 
        cbc_predic[idx] = cbc.numpy() 
        gap_predic[idx] = gap.numpy() 
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

    filename = f"{root_folder}results/predictions_{model_name}{subversion}.npz"

    # Save data
    # print(dns_target)
    np.savez(
        filename,
        lr_target=lr_target,
        hr_target=hr_target,
        hr_predic=hr_predic,
        fl_target=fl_target,
        dns_target=dns_target,
        cbc_predic=cbc_predic,
        gap_predic=gap_predic,
        xlr=xlr,
        ylr=ylr,
        xhr=xhr,
        yhr=yhr
    )

    filename = f"{root_folder}results/predictions_us{us:02d}_{model_name}{subversion}.mat"

    # # Save data

    sio.savemat(
        filename,
        {'lr_target' : lr_target,
        'hr_target' : hr_target,
        'hr_predic' : hr_predic,
        'fl_target' : fl_target,
        'dns_target': dns_target,
        'cbc_predic': cbc_predic,
        'gap_predic': gap_predic,
        'xlr': xlr,
        'ylr': ylr,
        'xhr': xhr,
        'yhr': yhr}
    )

    return
