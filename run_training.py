# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 11:38:43 2021
@author: guemesturb
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
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

    nx, ny, _, channels, _, _, _ = get_conf(args.case)

    """
        Generate data pipelines
    """

    # Call pipeline generator

    dataset_train, dataset_valid = generate_pipeline_training(root_folder, us, channels, noise, validation_split, shuffle_buffer, batch_size, n_prefetch)
    
    """
        Generate training models
    """

    # Define model class for Tensorflow architectures

    ganpiv = GANPIV('test', us, nx, ny, channels=channels)

    # Generate model and loss objects for desired arquitecture

    generator, discriminator, generator_loss, discriminator_loss = ganpiv.architecture01()
    
    # Generate optimizers

    generator_optimizer, discriminator_optimizer = ganpiv.optimizer(learning_rate)

    """
        Run training loop
    """

    # Call training loop
    
    training_loop(root_folder, model_name, dataset_train, dataset_valid, generator, discriminator, generator_loss, discriminator_loss, generator_optimizer, discriminator_optimizer, pretrained, epochs, saving_freq)

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
    parser.add_argument("-v", "--validation_split", type=float, default=0.2)
    args = parser.parse_args()

    """
        Define case options
    """

    us = args.upsampling                                               # Subsampling case
    root_folder = f'/STORAGE01/aguemes/gan-piv/{args.case}/ss{us:02}/' # Folder containing the data for the selected case
    model_name = args.model_name                                       # String containing the nomel name
    noise = f"{args.noise:03d}"
    validation_split = args.validation_split
    learning_rate = args.learning_rate                                 # Learning rate for the SGD algorithm

    """
        Define training options
    """

    epochs = 100                    # Number of epochs for the training
    n_prefetch = 4                  # Number of tfrecords files prefeteched during the training
    batch_size = 8                  # Batch size dimension for the stochastic gradient descent (SGD) algorithm
    saving_freq = 5                 # Saving frequency during the training loop
    pretrained = False              # Flag to indicate if the model needs to be trained from a previous trained state or not
    learning_rate = 1e-4            # Learning rate for the SGD algorithm
    shuffle_buffer = 250            # Number of files shuffled inside each tfrecord (for a proper shuffeling, at least as large as the number of files inside each tfrecords)
    
    """
        Run execution logic
    """

    main()

