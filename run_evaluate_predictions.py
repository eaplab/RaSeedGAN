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

                root_folder = f'/STORAGE01/aguemes/gan-piv/{case}/ss{us:02}/' # Folder containing the data for the selected case

                """
                    Scale data
                """

                nx, ny, res, channels, n_samples_test, n_samples_train, _ = get_conf(case)

                # Define path to file containing scaling value

                filename = f"{root_folder}tfrecords/scaling_us{us}_noise_{noise}.npz"

                if channels == 2:

                    Upiv_mean = (np.expand_dims(np.load(filename)['Upiv_mean'], axis=0)) *res#/ 512 / 0.013
                    Vpiv_mean = (np.expand_dims(np.load(filename)['Vpiv_mean'], axis=0)) *res#/ 512 / 0.013
                    Uptv_mean = (np.expand_dims(np.load(filename)['Uptv_mean'], axis=0)) *res#/ 512 / 0.013
                    Vptv_mean = (np.expand_dims(np.load(filename)['Vptv_mean'], axis=0)) *res#/ 512 / 0.013

                    Upiv_std = (np.expand_dims(np.load(filename)['Upiv_std'], axis=0)) *res#/ 512 / 0.013
                    Vpiv_std = (np.expand_dims(np.load(filename)['Vpiv_std'], axis=0)) *res#/ 512 / 0.013
                    Uptv_std = (np.expand_dims(np.load(filename)['Uptv_std'], axis=0)) *res#/ 512 / 0.013
                    Vptv_std = (np.expand_dims(np.load(filename)['Vptv_std'], axis=0)) *res#/ 512 / 0.013
                
                    filename = f"{root_folder}results/predictions_{model_name}.npz"

                    data = np.load(filename)

                    dns_target = data['dns_target'] * res
                    cbc_predic = data['cbc_predic'] * res
                    hr_predic  = data['hr_predic'] 
                    hr_target  = data['hr_target'] 
                    lr_target  = data['lr_target'] 
                    fl_target  = data['fl_target']

                    """
                        Error metrics
                    """

                    # Mean-squared error
                    print(np.sqrt(np.nanmean(np.divide((dns_target[:, :, 1:, 0] -  hr_predic[:, :, 1:, 0])**2, Uptv_std[:, :,1:] * Uptv_std[:, :,1:], out=np.zeros_like(dns_target[:, :, 1:, 0]), where=Uptv_std[:, :,1:]!=0))))
                    print(np.sqrt(np.nanmean(np.divide((dns_target[:, :, :, 0] -  hr_predic[:, :, :, 0])**2, Uptv_std * Uptv_std, out=np.zeros_like(dns_target[:, :, :, 0]), where=Uptv_std!=0))))
                    print(np.sqrt(np.nanmean(np.divide((dns_target[:, :, :, 1] -  hr_predic[:, :, :, 1])**2, Vptv_std * Vptv_std, out=np.zeros_like(dns_target[:, :, :, 0]), where=Vptv_std!=0))))
                    print(np.sqrt(np.nanmean(np.divide((dns_target[:, :, :, 0] - cbc_predic[:, :, :, 0])**2, Uptv_std * Uptv_std, out=np.zeros_like(dns_target[:, :, :, 0]), where=Uptv_std!=0))))
                    print(np.sqrt(np.nanmean(np.divide((dns_target[:, :, :, 1] - cbc_predic[:, :, :, 1])**2, Vptv_std * Vptv_std, out=np.zeros_like(dns_target[:, :, :, 0]), where=Vptv_std!=0))))


                    import matplotlib.pyplot as plt

                    plt.subplot(411)
                    plt.imshow( lr_target[90, :, :, 0]-Upiv_mean[0,:,:], vmin=-0.3,vmax=0.3, cmap='RdBu_r', extent=[0,2,0,1])
                    plt.subplot(412)
                    plt.imshow(dns_target[90, :, :, 0]-Uptv_mean[0,:,:], vmin=-0.3,vmax=0.3, cmap='RdBu_r', extent=[0,2,0,1])
                    plt.subplot(413)
                    plt.imshow(cbc_predic[90, :, :, 0]-Uptv_mean[0,:,:], vmin=-0.3,vmax=0.3, cmap='RdBu_r', extent=[0,2,0,1])
                    plt.subplot(414)
                    plt.imshow( hr_predic[90, :, :, 0]-Uptv_mean[0,:,:], vmin=-0.3,vmax=0.3, cmap='RdBu_r', extent=[0,2,0,1])

                    plt.savefig('test.png')


                elif channels == 1:

                    Tpiv_mean = (np.expand_dims(np.load(filename)['Tpiv_mean'], axis=0)) *res#/ 512 / 0.013
                    Tptv_mean = (np.expand_dims(np.load(filename)['Tptv_mean'], axis=0)) *res#/ 512 / 0.013

                    Tpiv_std = (np.expand_dims(np.load(filename)['Tpiv_std'], axis=0)) *res#/ 512 / 0.013
                    Tptv_std = (np.expand_dims(np.load(filename)['Tptv_std'], axis=0)) *res#/ 512 / 0.013
                
                    filename = f"{root_folder}results/predictions_{model_name}.npz"

                    data = np.load(filename)

                    dns_target = data['dns_target'] * res
                    cbc_predic = data['cbc_predic'] * res
                    hr_predic  = data['hr_predic'] 
                    hr_target  = data['hr_target'] 
                    lr_target  = data['lr_target'] 
                    fl_target  = data['fl_target']

                    """
                        Error metrics
                    """

                    # Mean-squared error
                    print(np.sqrt(np.nanmean(np.divide((dns_target[:, :, :, 0] - hr_predic[:, :, :, 0])**2, Tptv_std * Tptv_std, out=np.zeros_like(dns_target[:, :, :, 0]), where=Tptv_std!=0))))
                    print(np.sqrt(np.nanmean(np.divide((dns_target[:, :, :, 0] - cbc_predic[:, :, :, 0])**2, Tptv_std * Tptv_std, out=np.zeros_like(dns_target[:, :, :, 0]), where=Tptv_std!=0))))
                    # print(np.sqrt(np.nanmean((dns_target[801:1000, :, :, 0] -  hr_predic[801:1000, :, :, 0])**2 / (Tptv_std * Tptv_std))))
                    # print(np.sqrt(np.nanmean((dns_target[801:1000, :, :, 0] - cbc_predic[801:1000, :, :, 0])**2 / (Tptv_std * Tptv_std))))


                    import matplotlib.pyplot as plt

                    plt.subplot(311)
                    plt.imshow(dns_target[990, :, :, 0], vmin=-0.02,vmax=0.02, cmap='RdBu_r', extent=[-2,6,-3,3])

                    plt.subplot(312)
                    plt.imshow(cbc_predic[990, :, :, 0], vmin=-0.02,vmax=0.02, cmap='RdBu_r', extent=[-2,6,-3,3])
                    plt.subplot(313)
                    plt.imshow(hr_predic[990, :, :, 0], vmin=-0.02,vmax=0.02, cmap='RdBu_r', extent=[-2,6,-3,3])

                    plt.savefig('test.png')

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
    args = parser.parse_args()
    noise = f"{args.noise:03d}"

    """
        Run execution logic
    """

    main()