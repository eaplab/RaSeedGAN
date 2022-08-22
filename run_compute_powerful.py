# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:02:25 2021
@author: guemesturb
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'
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
import argparse
from libs import *


def main():

    """
        Main execution logic
    """

    """
        Import case chracteristics
    """

    nx, ny, res, channels, _, _, _ = get_conf(args.case)
    
    """
        Generate data pipelines
    """

    # Call testing pipeline generator

    dataset_test, n_samp = generate_pipeline_testing(root_folder, us, channels, noise, subversion)

    """
        Generate training models
    """

    # Define model class for channel architectures

    channel = GANPIVpowerful(model_name, us, nx, ny, channels=channels)
    
    # Generate model and loss objects for desired arquitecture

    generator, discriminator, generator_loss, discriminator_loss = channel.build_architecture()

    # Generate optimizers

    generator_optimizer, discriminator_optimizer = channel.optimizer(learning_rate)

    """
        Compute predictions
    """

    compute_predictions(root_folder, model_name, dataset_test, n_samp, nx, ny, res, us, channels, noise, generator, discriminator, generator_optimizer, discriminator_optimizer, subversion)


    return

if __name__ == '__main__':

    """
        Parsing arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--case", type=str, required=True)
    parser.add_argument("-u", "--upsampling", type=int, required=True)
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("-n", "--noise", type=int, required=True)
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-s", "--subversion", type=str, default="")
    args = parser.parse_args()

    """
        Define case options
    """

    us = args.upsampling                                               # Subsampling case
    root_folder = f'/storage/aguemes/gan-piv/{args.case}/ss{us:02}/' # Folder containing the data for the selected case
    time_folder = f'/STORAGE01/aguemes/gan-piv/{args.case}/time-resolved/ss{us:02}/'
    model_name = args.model_name                                       # String containing the nomel name
    noise = f"{args.noise:03d}"
    learning_rate = args.learning_rate  
    subversion = args.subversion                                 # Learning rate for the SGD algorithm

    """
        Run execution logic
    """

    main()