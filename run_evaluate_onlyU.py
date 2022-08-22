# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 23:38:13 2021
@author: guemesturb
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
availale_GPUs = len(physical_devices) 
print('Using TensorFlow version: ', tf.__version__, ', GPU:', availale_GPUs)
print('Using Keras version: ', tf.keras.__version__)
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import sys
import argparse
import numpy as np
import tensorflow as tf
from libs import *
import scipy.io as sio
import matplotlib.pyplot as plt


def main():

    """
        Main execution logic
    """

    for case in list(args.case):

        for model_name in args.model_name:

            for us in args.upsampling:

                """
                    Display options
                """

                print("\n")
                print("--------------------------------------------")
                print(f"Case:\t\t\t {case}")
                print(f"Architecture:\t\t {model_name}")
                print(f"Upsampling factor:\t x{us}")
                print("\n")

                """
                    Define case options
                """

                root_folder = f'/storage/aguemes/gan-piv/{case}/ss{us:02}/' # Folder containing the data for the selected case

                """
                    Scale data
                """

                nx, ny, res, channels, n_samples_test, n_samples_train, _ = get_conf(case)

                # Define path to file containing scaling value

                filename = f"{root_folder}tfrecords/scaling_us{us}_noise_{noise}.npz"

                if channels == 2:

                    Upiv_mean = (np.expand_dims(np.load(filename)['Upiv_mean'], axis=0)) *res#/ 512 / 0.013
                    Uptv_mean = (np.expand_dims(np.load(filename)['Uptv_mean'], axis=0)) *res#/ 512 / 0.013
                    
                    Upiv_std = (np.expand_dims(np.load(filename)['Upiv_std'], axis=0)) *res#/ 512 / 0.013
                    Uptv_std = (np.expand_dims(np.load(filename)['Uptv_std'], axis=0)) *res#/ 512 / 0.013
                    
                    filename = f"{root_folder}results/predictions_{model_name}{subversion}.npz"

                    data = np.load(filename)

                    dns_target = data['dns_target'] * res
                    cbc_predic = data['cbc_predic'] * res
                    gap_predic = data['gap_predic'] * res
                    hr_predic  = data['hr_predic'] 
                    hr_target  = data['hr_target'] 
                    lr_target  = data['lr_target'] 
                    fl_target  = data['fl_target']

                    # dns_target = np.where(np.sum(fl_target, axis=0) == 0, np.nan, dns_target)
                    # cbc_predic = np.where(np.sum(fl_target, axis=0) == 0, np.nan, cbc_predic)
                    # gap_predic = np.where(np.sum(fl_target, axis=0) == 0, np.nan, gap_predic)
                    # hr_predic = np.where(np.sum(fl_target, axis=0) == 0, np.nan, hr_predic)

                    scaU = np.nanvar(dns_target[:,:,:,0])
                    
                    """
                        Error metrics
                    """

                    # Mean-squared error
                    
                    print('GAN')
                    # print(np.sqrt(np.nanmean((dns_target[:, :, :, 0] -  hr_predic[:, :, :, 0])**2/ scaU)))
                    # print(np.sqrt(np.nanmean((dns_target[:, :, :, 1] -  hr_predic[:, :, :, 1])**2/ scaU)))
                    print(np.sqrt(np.nanmean((dns_target[:, us:-us, us:-us, 0] -  hr_predic[:, us:-us, us:-us, 0])**2/ scaU)))
                    # print(np.sqrt(np.nanmean((dns_target[:, 20:76, 75:113, 0] -  hr_predic[:, 20:76, 75:113, 0])**2/ scaU)))
                    # print(np.sqrt(np.nanmean((dns_target[:, 20:76, 75:113, 1] -  hr_predic[:, 20:76, 75:113, 1])**2/ scaU)))
                    print('Cubic')
                    # print(np.sqrt(np.nanmean((dns_target[:, :, :, 0] -  cbc_predic[:, :, :, 0])**2/ scaU)))
                    # print(np.sqrt(np.nanmean((dns_target[:, :, :, 1] -  cbc_predic[:, :, :, 1])**2/ scaU)))
                    print(np.sqrt(np.nanmean((dns_target[:, us:-us, us:-us, 0] -  cbc_predic[:, us:-us, us:-us, 0])**2/ scaU)))
                    # print(np.sqrt(np.nanmean((dns_target[:, 20:76, 75:113, 0] -  cbc_predic[:, 20:76, 75:113, 0])**2/ scaU)))
                    # print(np.sqrt(np.nanmean((dns_target[:, 20:76, 75:113, 1] -  cbc_predic[:, 20:76, 75:113, 1])**2/ scaU)))
                    print('Gappy')
                    # print(np.sqrt(np.nanmean((dns_target[:, :, :, 0] -  gap_predic[:, :, :, 0])**2/ scaU)))
                    # print(np.sqrt(np.nanmean((dns_target[:, :, :, 1] -  gap_predic[:, :, :, 1])**2/ scaU)))
                    print(np.sqrt(np.nanmean((dns_target[:, us:-us, us:-us, 0] -  gap_predic[:, us:-us, us:-us, 0])**2/ scaU)))
                    # print(np.sqrt(np.nanmean((dns_target[:, 20:76, 75:113, 0] -  gap_predic[:, 20:76, 75:113, 0])**2/ scaU)))
                    # print(np.sqrt(np.nanmean((dns_target[:, 20:76, 75:113, 1] -  gap_predic[:, 20:76, 75:113, 1])**2/ scaU)))

                    # plt.subplot(141)
                    # plt.imshow( hr_predic[90, :, :, 0], vmin=-2,vmax=2, cmap='RdBu_r', extent=[0,6,-1,1])
                    # plt.subplot(142)
                    # plt.imshow(dns_target[90, :, :, 0], vmin=-2,vmax=2, cmap='RdBu_r', extent=[0,6,-1,1])
                    # plt.subplot(143)
                    # plt.imshow(cbc_predic[90, :, :, 0], vmin=-2,vmax=2, cmap='RdBu_r', extent=[0,6,-1,1])
                    # plt.subplot(144)
                    # plt.imshow(gap_predic[90, :, :, 0], vmin=-2,vmax=2, cmap='RdBu_r', extent=[0,6,-1,1])
                    # plt.savefig('testOnlyU.png')
    return


if __name__ == '__main__':

    """
        Parsing arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--case", type=str, nargs='+', required=True)
    parser.add_argument("-m", "--model_name", type=str, nargs='+', required=True)
    parser.add_argument("-u", "--upsampling", type=int, nargs='+', required=True)
    parser.add_argument("-n", "--noise", type=int, required=True)
    parser.add_argument("-s", "--subversion", type=str, default="")
    args = parser.parse_args()
    noise = f"{args.noise:03d}"
    subversion = args.subversion   

    """
        Run execution logic
    """

    main()