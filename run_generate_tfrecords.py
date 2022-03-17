# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 18:37:16 2021
@author: guemesturb
"""

import argparse
from libs import *


def main():

    """
        Main execution logic
    """

    """
        Import case chracteristics
    """

    nx, ny, res, channels, n_samples_test, n_samples_train, max_samples_per_tf = get_conf(args.case)

    """
        Generate training tfrecords
    """

    generate_tfrecords_training(root_folder, us, n_samples_train, max_samples_per_tf, channels, noise)

    """
        Generate testing tfrecords
    """

    generate_tfrecords_testing(root_folder, us, n_samples_train, n_samples_test, max_samples_per_tf, channels, noise)

    """
        Generate scaling data
    """

    generate_scaling_data(root_folder, nx, ny, us, n_samples_train, channels, noise)

    return


if __name__ == '__main__':

    """
        Parsing arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--case", type=str, required=True)
    parser.add_argument("-u", "--upsampling", type=int, required=True)
    parser.add_argument("-n", "--noise", type=int, required=True)
    args = parser.parse_args()

    """
        Define case options
    """

    us = args.upsampling                                               # Subsampling case
    root_folder = f'data/{args.case}/'           # Folder containing the data for the selected case
    noise = f"{args.noise:03d}"

    """
        Run execution logic
    """

    main()