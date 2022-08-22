# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 20:41:41 2021
@author: guemesturb
"""


def get_conf(case):
    """
        Function to retrieve the case characteristics.

    :param case: String indicating the selected case.
    :return nx: Integer indicating the grid points of the selected case in the streamwise direction.
    :return nx: Integer indicating the grid points of the selected case in the wall-normal direction.
    :return n_samples_test: Integer indicating the number of samples to be used in the testing dataset.
    :return n_samples_train: Integer indicating the number of samples to be used in the training dataset.
    :return max_samples_per_tf: Integer indicating the maximum number of samples to store per tfrecord file
    """

    if case == 'channel':

        nx = 64                 # Grid points in the streamwise direction for the low resolution data
        ny = 32                 # Grid points in the wall-normal direction for the low resolution data
        res = 1/512/0.013       # Resolution
        channels = 2            # Number of output channels
        n_samples_test = 1753 #1856   # Number of testing samples
        n_samples_train = 10000 # Number of training samples
        max_samples_per_tf = 50 # Maximum number of samples per tfrecord file

    if case == 'pinball':

        nx = 36                 # Grid points in the streamwise direction for the low resolution data
        ny = 12                 # Grid points in the wall-normal direction for the low resolution data
        res = 1                 # Resolution
        channels = 2            # Number of output channels
        n_samples_test = 737    # Number of testing samples
        n_samples_train = 4000  # Number of training samples
        max_samples_per_tf = 50 # Maximum number of samples per tfrecord file

    if case == 'exptbl':

        nx = 31                 # Grid points in the streamwise direction for the low resolution data
        ny = 32                 # Grid points in the wall-normal direction for the low resolution data
        res = 1                # Resolution
        channels = 2
        n_samples_test = 2000   # Number of testing samples
        n_samples_train = 30000 # Number of training samples
        max_samples_per_tf = 1000 # Maximum number of samples per tfrecord file

    if case == 'sst':

        nx = 90                 # Grid points in the streamwise direction for the low resolution data
        ny = 45                 # Grid points in the wall-normal direction for the low resolution data
        res = 1                # Resolution
        channels = 1            # Number of output channels
        n_samples_test = 1305   # Number of testing samples
        n_samples_train = 6000 # Number of training samples
        max_samples_per_tf = 50 # Maximum number of samples per tfrecord file

    if case == 'buoyancy':

        nx = 64                 # Grid points in the streamwise direction for the low resolution data
        ny = 64                 # Grid points in the wall-normal direction for the low resolution data
        res = 1                 # Resolution
        channels = 1            # Number of output channels
        n_samples_test = 3000   # Number of testing samples
        n_samples_train = 12000 # Number of training samples
        max_samples_per_tf = 50 # Maximum number of samples per tfrecord file

    return nx, ny, res, channels, n_samples_test, n_samples_train, max_samples_per_tf