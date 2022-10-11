# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 11:38:43 2021
@author: guemesturb
"""


import os
import re
import h5py
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tqdm import tqdm


def generate_pipeline_testing(root_folder, us, channels, noise, subversion=""):
    """
        Function to generate Tensorflow data pipeline for the testing dataset.

    :param root_folder:         String indicating the folder where the data is stored.
    :return dataset_test:       Tensorflow object containing the pipeline for the testing dataset.
    :return tot_samples_per_ds: Integer showing the total number of samples in the dataset
    """

    """
        Prepare files
    """

    # Define path where the tfrecord files are stored

    tfr_path = f"{root_folder}tfrecords/test/"

    # Select files in the tfrecord folder

    tfr_files = sorted([os.path.join(tfr_path,f) for f in os.listdir(tfr_path) if os.path.isfile(os.path.join(tfr_path,f))])
    
    # Keep only files terminated on .tfrecords

    regex = re.compile(f'.tfrecords')

    tfr_files = ([string for string in tfr_files if re.search(regex, string)])

    # Select specific noise level

    regex = re.compile(f'noise_{noise}')

    tfr_files = ([string for string in tfr_files if re.search(regex, string)])

    regex = re.compile(f'LowerDensity')

    tfr_files = ([string for string in tfr_files if not re.search(regex, string)])
    
    regex = re.compile(f'lowV')

    tfr_files = ([string for string in tfr_files if not re.search(regex, string)])
    
    # Find number of samples for each tfrecord file
    
    if root_folder.__contains__('exptbl'):

        n_samples_per_tfr = np.array([int(s.split('.')[-2][-4:].replace('_', '')) for s in tfr_files])
        n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]

    else:

        n_samples_per_tfr = np.array([int(s.split('.')[-2][-2:].replace('_', '')) for s in tfr_files])
        n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]

    # Define total number of samples in the tfrecord files

    tot_samples_per_ds = sum(n_samples_per_tfr)

    # Initialize the testing dataset with the assigned tfrecord files. Internal shuffle is not applied

    tfr_files_test_ds = tf.data.Dataset.list_files(tfr_files, shuffle=False)

    # Parse the tfrecord files assigned to the testing dataset

    tfr_files_test_ds = tf.data.TFRecordDataset(tfr_files_test_ds)

    """
        Generated testing dataset pipeline
    """

    # Parse the information contained in the tfrecord files

    dataset_test = tfr_files_test_ds.map(lambda x: tf_parser_testing(x, root_folder, us, channels, noise), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset_test, tot_samples_per_ds


def generate_pipeline_training(root_folder, us, channels, noise, validation_split=0.2, shuffle_buffer=200, batch_size=8, n_prefetch=4):
    """
        Function to generate Tensorflow data pipelines for the training and validation datasets.

    :param root_folder:      String indicating the folder where the data is stored.
    :param validation_split: Float indicating the percentage of the training dataset to be used as validation dataset. Default is 0.2.
    :param shuffle_buffer:   Integer idicating the number of times to shuffle a file inside each tfrecord. Default is 200.
    :param batch_size:       Integer indicating the number of sample to be used during the stochastic gradient descent fitting of the neural network weights. Default is 8.
    :param n_prefecth:       Integer indicating the number of files preallocated during the training loop for each tfrecord. Default is 4.
    :return dataset_train:   Tensorflow object containing the pipeline for the training dataset.
    :return dataset_valid:   Tensorflow object containing the pipeline for the validation dataset.
    """

    """
        Prepare files
    """

    # Define path where the tfrecord files are stored

    tfr_path = f"{root_folder}tfrecords/train/"

    # Select files in the tfrecord folder

    tfr_files = sorted([os.path.join(tfr_path,f) for f in os.listdir(tfr_path) if os.path.isfile(os.path.join(tfr_path,f))])

    # Keep only files terminated on .tfrecords

    regex = re.compile(f'.tfrecords')

    tfr_files = ([string for string in tfr_files if re.search(regex, string)])

    # Select specific noise level

    regex = re.compile(f'noise_{noise}')

    tfr_files = ([string for string in tfr_files if re.search(regex, string)])
    regex = re.compile(f'LowerDensity')

    tfr_files = ([string for string in tfr_files if not re.search(regex, string)])
    regex = re.compile(f'lowV')

    tfr_files = ([string for string in tfr_files if not re.search(regex, string)])
    
    """
        Separate files for training and validation
    """

    # Find number of samples for each tfrecord file
    
    n_samples_per_tfr = np.array([int(s.split('.')[-2][-2:]) for s in tfr_files])
    n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]

    # Define the cumulative number of samples in the tfrecord files

    cumulative_samples_per_tfr = np.cumsum(np.array(n_samples_per_tfr))

    # Define total number of samples in the tfrecord files

    tot_samples_per_ds = sum(n_samples_per_tfr)

    # Define number of tfrecord files available

    n_tfr_loaded_per_ds = int(tfr_files[0].split('_')[-3][-3:])

    # Ensure that the selected number of tfrecord files match the defined number of files available

    tfr_files = [string for string in tfr_files if int(string.split('_')[-3][:3]) <= n_tfr_loaded_per_ds]

    # Computer the number of files to be used in the training dataset

    n_samp_train = int(sum(n_samples_per_tfr) * (1 - validation_split))

    # Compute the number of file to be used in the validation dataset

    n_samp_valid = sum(n_samples_per_tfr) - n_samp_train

    # Define number of tfrecord files to be used for the training dataset and the  remaining number of files to be added from a shared tfrecord file with the validation dataset
    
    (n_files_train, samples_train_left) = np.divmod(n_samp_train, n_samples_per_tfr[0])

    # If there is shared files, add one to the number of training dataset files

    if samples_train_left > 0:

        n_files_train += 1

    # Select the tfrecord files for the training dataset

    tfr_files_train = [string for string in tfr_files if int(string.split('_')[-3][:3]) <= n_files_train]
    
    # Select number of tfrecord files that shared files between validation and training dataset (this is done because it is not mandatory that each tfrecord contains the same number of samples)

    n_tfr_left = np.sum(np.where(cumulative_samples_per_tfr < samples_train_left, 1, 0)) + 1
    
    # Define shared tfrecord files

    if sum([int(s.split('.')[-2][-2:]) for s in tfr_files_train]) != n_samp_train:

        # If the tfrecord files selected for the training dataset contains more samples than the number of training samples the last tfrecord file is shared with the validation dataset 
        shared_tfr = tfr_files_train[-1]
        tfr_files_valid = [shared_tfr]
    else:
        # If not, the validation dataset is initialized empty
        shared_tfr = ''
        tfr_files_valid = list()

    # The rest of tfrecord files not selected for the training dataset are added to the validation dataset

    tfr_files_valid.extend([string for string in tfr_files if string not in tfr_files_train])

    # Sort tfrecord files in the validation dataset
    tfr_files_valid = sorted(tfr_files_valid)

    # The shared tfrecord file is initialized as a TF constant

    shared_tfr_out = tf.constant(shared_tfr)

    # Initialize the number of tfrecord files as a TF constant

    n_tfr_per_ds = tf.constant(n_tfr_loaded_per_ds)

    # Initilize list containing the number of samples contained in each tfrecord file

    n_samples_loaded_per_tfr = list()

    # Assign files to the list

    if n_tfr_loaded_per_ds>1:

        # If there is more than one tfrecord file, the number of samples for each tfrecord file is added to the list except the last file

        n_samples_loaded_per_tfr.extend(n_samples_per_tfr[:n_tfr_loaded_per_ds-1])
        
        # The number of samples for the last tfrecord file is added as the difference between the toal number of samples and the cumulative number of samples until the penultimate file (REDUNDANT, it should be eliminated)

        n_samples_loaded_per_tfr.append(tot_samples_per_ds - cumulative_samples_per_tfr[n_tfr_loaded_per_ds-2])

    else:

        # If there is only one tfrecord file, add the total number of samples

        n_samples_loaded_per_tfr.append(tot_samples_per_ds)

    # Convert the list containing the number of samples per tfrecord file into a Numpy array

    n_samples_loaded_per_tfr = np.array(n_samples_loaded_per_tfr)

    # Initialize the training dataset with the assigned tfrexcord files. Seed is added to shuffle internally the tfrecord files
    
    tfr_files_train_ds = tf.data.Dataset.list_files(tfr_files_train, seed=666)

    # Initialize the validation dataset with the assigned tfrexcord files. Seed is added to shuffle internally the tfrecord files

    tfr_files_val_ds = tf.data.Dataset.list_files(tfr_files_valid, seed=686)

    # Check the number of shared tfrecord files

    if n_tfr_left>1:

        # If it is more than one, generate variable with the number of files in the shared tfrecord files for the training dataset

        samples_train_shared = samples_train_left - cumulative_samples_per_tfr[n_tfr_left-2]

        # Define the total number of samples in the shared tfrecord file

        n_samples_tfr_shared = n_samples_loaded_per_tfr[n_tfr_left-1]

    else:

        # If it is only one, rename the variable containing the files in the shared tfrecord file for the training dataset

        samples_train_shared = samples_train_left

        # Define the total number of samples in the shared tfrecord file

        n_samples_tfr_shared = n_samples_loaded_per_tfr[0]

    # Parse the tfrecord files assigned to the training dataset, excluding the samples belonging to the validation dataset

    # If the lambda file x, corresponding to the tfrecord file name, is equal to the shared tfrecord file name, take from that tfrecord file the first N number of samples, being N the number of samples in the shared tfrecord file assigned to the training dataset
    
    # If the lambda file x, corresponding to the tfrecord file name, contains samples belonging only to the training dataset, take the number of samples contained in that tfrecord file (check with the tfrecord file name that contains the information)

    tfr_files_train_ds = tfr_files_train_ds.interleave(
        lambda x : tf.data.TFRecordDataset(x).take(samples_train_shared) if tf.math.equal(x, shared_tfr_out) else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-3],sep='-')[0], tf.int32)-1)), 
        cycle_length=16, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Parse the tfrecord files assigned to the validation dataset, excluding the samples belonging to the training dataset

    # If the lambda file x, corresponding to the tfrecord file name, is equal to the shared tfrecord file name, take from that tfrecord file the last M number of samples, being M the difference between the total number of samples in the shared tfrecord file and the number of samples in the shared tfrecord file assigned to the training dataset

    # If the lambda file x, corresponding to the tfrecord file name, contains samples belonging only to the training dataset, take the number of samples contained in that tfrecord file (check with the tfrecord file name that contains the information)

    tfr_files_val_ds = tfr_files_val_ds.interleave(
        lambda x : tf.data.TFRecordDataset(x).skip(samples_train_shared).take(n_samples_tfr_shared - samples_train_shared) if tf.math.equal(x, shared_tfr_out) else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-3],sep='-')[0], tf.int32)-1)),
        cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    """
        Generated training dataset pipeline
    """

    # Parse the information contained in the tfrecord files

    dataset_train = tfr_files_train_ds.map(lambda x: tf_parser(x, root_folder, us, channels, noise), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle the samples

    dataset_train = dataset_train.shuffle(shuffle_buffer)

    # Provide the pipeline with the batch size information

    dataset_train = dataset_train.batch(batch_size=batch_size)

    # Provide the pipeline with the number of prefetched files information

    dataset_train = dataset_train.prefetch(n_prefetch)

    """
        Generated validation dataset pipeline
    """

    # Parse the information contained in the tfrecord files

    dataset_valid = tfr_files_val_ds.map(lambda x: tf_parser(x, root_folder, us, channels, noise), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle the samples

    dataset_valid = dataset_valid.shuffle(shuffle_buffer)

    # Provide the pipeline with the batch size information

    dataset_valid = dataset_valid.batch(batch_size=batch_size)

    # Provide the pipeline with the number of prefetched files information
    
    dataset_valid = dataset_valid.prefetch(n_prefetch)

    return dataset_train, dataset_valid


def generate_scaling_data(root_folder, nx, ny, us, n_samples_train, channels, noise):
    """
        Function to generate the scaling values from the training dataset.
    
    :param root_folder:        String indicating the folder where the data is stored.
    :param nx:                 Integer containing the grid points in the streamwise direction for the low-resolution data.
    :param ny:                 Integer containing the grid points in the wall-normal direction for the low-resolution data.
    :param us:                 Integer containing the upsampling ratio between the low- and high-resolution data.
    :param n_samples_train:    Integer indicating the number of samples in the training dataset.
    """


    np.seterr(divide='ignore', invalid='ignore')

    if channels == 2:

        Upiv = np.zeros((n_samples_train, ny,    nx))
        Vpiv = np.zeros((n_samples_train, ny,    nx))
        Uptv = np.zeros((n_samples_train, ny*us, nx*us))
        Vptv = np.zeros((n_samples_train, ny*us, nx*us))
        Flag = np.zeros((n_samples_train, ny*us, nx*us))

        # Iterate over the number of samples

        for idx in tqdm(range(n_samples_train)):

            # Define path to Matlab file containing the actual sample

            piv_path = f"{root_folder}ss01/piv_noise{noise}/SS1_{(idx+1):06d}.mat"
            ptv_path = f"{root_folder}ss{us:02d}/ptv_noise{noise}/SS{us}_{(idx+1):06d}.mat"
                
            # Load Matlab file

            piv = h5py.File(piv_path, 'r')
            ptv = h5py.File(ptv_path, 'r')

            # Write each variable in their corresponding global matrices

            Upiv[idx, :, :] = np.array(piv['Uptv']).T
            Vpiv[idx, :, :] = np.array(piv['Vptv']).T
            Uptv[idx, :, :] = np.array(ptv['Uptv']).T
            Vptv[idx, :, :] = np.array(ptv['Vptv']).T
            Flag[idx, :, :] = np.array(ptv['Flagptv']).T
        
        # Compute mean values

        Upiv_mean = np.mean(Upiv, axis=0)
        Vpiv_mean = np.mean(Vpiv, axis=0)
        Uptv_mean = np.sum(Uptv, axis=0) / np.sum(Flag, axis=0)
        Vptv_mean = np.sum(Vptv, axis=0) / np.sum(Flag, axis=0)

        # Compute standard deviation values
        
        Upiv_std = np.std(Upiv, axis=0)
        Vpiv_std = np.std(Vpiv, axis=0)
        Uptv_std = np.sqrt(np.sum((Uptv - Uptv_mean*Flag)**2, axis=0) / np.sum(Flag, axis=0))
        Vptv_std = np.sqrt(np.sum((Vptv - Vptv_mean*Flag)**2, axis=0) / np.sum(Flag, axis=0))
        

        # Define path to stored the scaling values

        filename = f"{root_folder}ss{us:02}/tfrecords/scaling_us{us}_noise_{noise}.npz"

        # Save scaling values

        np.savez(
            filename,
            Upiv_mean=Upiv_mean,
            Vpiv_mean=Vpiv_mean,
            Uptv_mean=Uptv_mean,
            Vptv_mean=Vptv_mean,
            Upiv_std=Upiv_std,
            Vpiv_std=Vpiv_std,
            Uptv_std=Uptv_std,
            Vptv_std=Vptv_std,
        )
    
    elif channels == 1:

        Tpiv = np.zeros((n_samples_train, ny,    nx))
        Tptv = np.zeros((n_samples_train, ny*us, nx*us))
        Flag = np.zeros((n_samples_train, ny*us, nx*us))

        # Iterate over the number of samples

        for idx in tqdm(range(n_samples_train)):

            # Define path to Matlab file containing the actual sample

            piv_path = f"{root_folder}ss01/piv_noise{noise}/SS1_{(idx+1):06d}.mat"
            ptv_path = f"{root_folder}ss{us:02d}/ptv_noise{noise}/SS{us}_{(idx+1):06d}.mat"
                
            # Load Matlab file

            try:

                piv = h5py.File(piv_path, 'r')
                ptv = h5py.File(ptv_path, 'r')

            except OSError:

                pass

            # Write each variable in their corresponding global matrices

            Tpiv[idx, :, :] = np.array(piv['T']).T
            Tptv[idx, :, :] = np.array(ptv['T']).T
            Flag[idx, :, :] = np.array(ptv['Flagptv']).T
        
        # Compute mean values

        Tpiv_mean = np.mean(Tpiv, axis=0)
        Tptv_mean = np.sum(Tptv, axis=0) / np.sum(Flag, axis=0)

        # Compute standard deviation values
        
        Tpiv_std = np.std(Tpiv, axis=0)
        Tptv_std = np.sqrt(np.sum((Tptv - Tptv_mean*Flag)**2, axis=0) / np.sum(Flag, axis=0))
        

        # Define path to stored the scaling values

        filename = f"{root_folder}ss{us:02}/tfrecords/scaling_us{us}_noise_{noise}.npz"

        # Save scaling values

        np.savez(
            filename,
            Tpiv_mean=Tpiv_mean,
            Tptv_mean=Tptv_mean,
            Tpiv_std=Tpiv_std,
            Tptv_std=Tptv_std,
        )

    return


def generate_tfrecords_testing(root_folder, us, n_samples_train, n_samples_test, max_samples_per_tf, channels, noise, subversion):
    """
        Function to generate tfrecords files for the training data.
    
    :param root_folder:        String indicating the folder where the data is stored.
    :param us:                 Integer containing the upsampling ratio between the low- and high-resolution data.
    :param n_samples_train:    Integer indicating the number of samples in the training dataset.
    :param max_samples_per_tf: Integer indicating the maximum number of samples that can be stored in a single tfrecord file.
    """
    

    """
        Store tfrecord files
    """

    # Define path where the tfrecord files will be stored

    tfr_path = f"{root_folder}ss{us:02}/tfrecords/test/"

    # Check if the tfrecord folder exists

    if not os.path.exists(tfr_path):

        # If not, create it

        os.makedirs(tfr_path)

    # Define base for the tfrecords filename

    tfrecords_filename_base = f"{tfr_path}us{us}_noise_{noise}{subversion}_"

    # Calculate number of tfrecord files that will be generated

    n_sets = int(np.ceil(n_samples_test / max_samples_per_tf))

    # Define index indicating where the training sample should be acquired from

    idx = n_samples_train

    # Iterate over the calculated number of tfrecord files

    for n_set in tqdm(range(1, n_sets + 1)):

        # Check if this is the last tfrecord file

        if n_set * max_samples_per_tf < n_samples_test:

            # If not, store the maximum number of samples per tfrecord file

            num_smp = max_samples_per_tf

        else:

            # If this is the last tfrecord file store the remaining samples, always equal or smaller than the maximum number of samples per tfrecord file

            num_smp = n_samples_test - (n_set - 1) * max_samples_per_tf

        # Define filename for the actual tfrecord file

        tfrecords_filename = tfrecords_filename_base + f"file_{n_set:03d}-of-{n_sets:03d}_samples_{num_smp}.tfrecords"

        # Open Tensorflow writer object

        writer = tf.io.TFRecordWriter(tfrecords_filename)

        # Iterate over the number of samples to be stored in the actual tfrecord file

        for i in range(num_smp):

            if channels == 2:

                piv_path = f"{root_folder}ss01/piv_noise{noise}{subversion}/SS1_{(idx+1):06d}.mat"
                ptv_path = f"{root_folder}ss{us:02d}/ptv_noise{noise}{subversion}/SS{us}_{(idx+1):06d}.mat"
                dns_path = f"{root_folder}ss{us:02d}/dns/SS{us}_{(idx+1):06d}.mat"
                cbc_path = f"{root_folder}ss{us:02d}/cubic_noise{noise}{subversion}/cubic_SS{us}_{(idx+1):06d}.mat"
                gap_path = f"{root_folder}ss{us:02d}/GappyPOD_SS{us}_noise{noise}{subversion}/GPOD_SS{us}_{(idx+1):06d}.mat"
                gap_path = f"{root_folder}ss{us:02d}/GappyPOD_SS{us}{subversion}_10000img/GPOD_SS{us}_{(idx+1):06d}.mat"
                
                if root_folder.__contains__('exptbl'):

                    piv = h5py.File(piv_path, 'r')
                    Upiv = np.array(piv['Uptv']).T[:,1:]
                    Vpiv = np.array(piv['Vptv']).T[:,1:]
                    
                    ptv = h5py.File(ptv_path, 'r')
                    Uptv = np.array(ptv['Uptv']).T[:, us:]
                    Vptv = np.array(ptv['Vptv']).T[:, us:]
                    Flag = np.array(ptv['Flagptv']).T[:, us:]
                    
                    dns = h5py.File(dns_path, 'r')

                    if us == 2:
                        Udns = np.array(dns['U']).T
                        Vdns = np.array(dns['V']).T
                        Udns = Udns[1::2, 4::2]
                        Vdns = Vdns[1::2, 4::2]

                    else:

                        Udns = np.array(dns['U']).T[:, us:]
                        Vdns = np.array(dns['V']).T[:, us:]

                    cbc = sio.loadmat(cbc_path)
                    Ucbc = np.array(cbc['U_interp'])[:, us:]
                    Vcbc = np.array(cbc['V_interp'])[:, us:]

                    gap = sio.loadmat(gap_path)
                    Ugap = np.array(gap['Uptv'])[:, us:]
                    Vgap = np.array(gap['Vptv'])[:, us:]

                else:
                    piv = h5py.File(piv_path, 'r')
                    Upiv = np.array(piv['Uptv']).T
                    Vpiv = np.array(piv['Vptv']).T
                    # piv = sio.loadmat(piv_path)
                    # Upiv = np.array(piv['Uptv'])
                    # Vpiv = np.array(piv['Vptv'])

                    
                    ptv = h5py.File(ptv_path, 'r')
                    Uptv = np.array(ptv['Uptv']).T
                    Vptv = np.array(ptv['Vptv']).T
                    Flag = np.array(ptv['Flagptv']).T
                    
<<<<<<< HEAD
                    # dns = h5py.File(dns_path, 'r')
                    dns = sio.loadmat(dns_path)
=======
                    dns = h5py.File(dns_path, 'r')
>>>>>>> d697111dba0b1b8df36000c0c120a92bf7e61e0e
                    Udns = np.array(dns['UDNS']).T
                    Vdns = np.array(dns['VDNS']).T
                    # dns = sio.loadmat(dns_path)
                    # Udns = np.array(dns['UDNS'])
                    # Vdns = np.array(dns['VDNS'])

                    cbc = sio.loadmat(cbc_path)
                    Ucbc = np.array(cbc['U_interp'])
                    Vcbc = np.array(cbc['V_interp'])
    
                    gap = sio.loadmat(gap_path)
                    Ugap = np.array(gap['Uptv'])[:, :]
                    Vgap = np.array(gap['Vptv'])[:, :]


                if idx == n_samples_train:

                    # If so, allocate variable for the grid resolution in the high- and low-resolution data

                    nx_piv = Upiv.shape[1]
                    ny_piv = Upiv.shape[0]

                    nx_ptv = Uptv.shape[1]
                    ny_ptv = Uptv.shape[0]

                    if root_folder.__contains__('exptbl'):


                        piv = h5py.File(f"{root_folder}ss01/piv_noise{noise}/SS1_{(idx+1):06d}.mat", 'r')

                        xlr = np.array(piv['XPIV']).T[:,1:]
                        ylr = np.array(piv['YPIV']).T[:,1:]

                        grid_path = f"{root_folder}ss{us:02d}/ptv_noise{noise}/SS{us}_grid.mat"
                        grid = sio.loadmat(grid_path) 

                        xhr = grid['X'][:, us:]
                        yhr = grid['Y'][:, us:]

                    else:

                        grid_path = f"{root_folder}ss01/piv_noise{noise}/SS1_grid.mat"
                        grid = sio.loadmat(grid_path)

                        xlr = np.array(grid['X']).T
                        ylr = np.array(grid['Y']).T

                        grid_path = f"{root_folder}ss{us:02d}/ptv_noise{noise}/SS{us}_grid.mat"

                        grid = sio.loadmat(grid_path)

                        xhr = grid['X']
                        yhr = grid['Y']

                # Define sample to be stored
                
                example = tf.train.Example(
                    features = tf.train.Features(
                        feature={
                            'i_sample': _int64_feature(idx),
                            'nx_piv': _int64_feature(int(nx_piv)),
                            'ny_piv': _int64_feature(int(ny_piv)),
                            'nx_ptv': _int64_feature(int(nx_ptv)),
                            'ny_ptv': _int64_feature(int(ny_ptv)),
                            'x_lr': _floatarray_feature(np.float32(xlr).flatten().tolist()),
                            'y_lr': _floatarray_feature(np.float32(ylr).flatten().tolist()),
                            'x_hr': _floatarray_feature(np.float32(xhr).flatten().tolist()),
                            'y_hr': _floatarray_feature(np.float32(yhr).flatten().tolist()),
                            'piv_raw1': _floatarray_feature(np.float32(Upiv).flatten().tolist()),
                            'piv_raw2': _floatarray_feature(np.float32(Vpiv).flatten().tolist()),
                            'ptv_raw1': _floatarray_feature(np.float32(Uptv).flatten().tolist()),
                            'ptv_raw2': _floatarray_feature(np.float32(Vptv).flatten().tolist()),
                            'dns_raw1': _floatarray_feature(np.float32(Udns).flatten().tolist()),
                            'dns_raw2': _floatarray_feature(np.float32(Vdns).flatten().tolist()),
                            'cbc_raw1': _floatarray_feature(np.float32(Ucbc).flatten().tolist()),
                            'cbc_raw2': _floatarray_feature(np.float32(Vcbc).flatten().tolist()),
                            'gap_raw1': _floatarray_feature(np.float32(Ugap).flatten().tolist()),
                            'gap_raw2': _floatarray_feature(np.float32(Vgap).flatten().tolist()),
                            'ptv_flag': _floatarray_feature(np.float32(Flag).flatten().tolist()),
                        }
                    )
                )  

            elif channels == 1:

                # Check if this is the first sample
            
                piv_path = f"{root_folder}ss01/piv_noise{noise}/SS1_{(idx+1):06d}.mat"
                ptv_path = f"{root_folder}ss{us:02d}/ptv_noise{noise}/SS{us}_{(idx+1):06d}.mat"
                dns_path = f"{root_folder}ss{us:02d}/dns/SS{us}_{(idx+1):06d}.mat"
                cbc_path = f"{root_folder}ss{us:02d}/cubic_noise{noise}/cubic_SS{us}_{(idx+1):06d}.mat"
                gap_path = f"{root_folder}ss{us:02d}/GappyPOD_SS{us}_noise{noise}/GPOD_SS{us}_{(idx+1):06d}.mat"
            
                piv = h5py.File(piv_path, 'r')
                ptv = h5py.File(ptv_path, 'r')
                if root_folder.__contains__('sst'):
                    Tpiv = np.array(piv['T']).T
                    Tptv = np.array(ptv['T']).T
                else:
                    Tpiv = np.array(piv['Tptv']).T
                    Tptv = np.array(ptv['Tptv']).T
                
                Flag = np.array(ptv['Flagptv']).T
                
                dns = sio.loadmat(dns_path)
                Tdns = np.array(dns['TDNS'])

                cbc = sio.loadmat(cbc_path)
                Tcbc = np.array(cbc['T_interp'])

                gap = sio.loadmat(gap_path)
                Tgap = np.array(gap['T'])[:, :]

                if idx == n_samples_train:

                    # If so, allocate variable for the grid resolution in the high- and low-resolution data

                    nx_piv = Tpiv.shape[1]
                    ny_piv = Tpiv.shape[0]

                    nx_ptv = Tptv.shape[1]
                    ny_ptv = Tptv.shape[0]

                    grid_path = f"{root_folder}ss01/piv_noise{noise}/SS1_grid.mat"
                    grid = sio.loadmat(grid_path)

                    xlr = np.array(grid['X']).T
                    ylr = np.array(grid['Y']).T

                    grid_path = f"{root_folder}ss{us:02d}/ptv_noise{noise}/SS{us}_grid.mat"

                    grid = sio.loadmat(grid_path)

                    xhr = grid['X']
                    yhr = grid['Y']

                # Define sample to be stored
                
                example = tf.train.Example(
                    features = tf.train.Features(
                        feature={
                            'i_sample': _int64_feature(idx),
                            'nx_piv': _int64_feature(int(nx_piv)),
                            'ny_piv': _int64_feature(int(ny_piv)),
                            'nx_ptv': _int64_feature(int(nx_ptv)),
                            'ny_ptv': _int64_feature(int(ny_ptv)),
                            'x_lr': _floatarray_feature(np.float32(xlr).flatten().tolist()),
                            'y_lr': _floatarray_feature(np.float32(ylr).flatten().tolist()),
                            'x_hr': _floatarray_feature(np.float32(xhr).flatten().tolist()),
                            'y_hr': _floatarray_feature(np.float32(yhr).flatten().tolist()),
                            'piv_raw1': _floatarray_feature(np.float32(Tpiv).flatten().tolist()),
                            'ptv_raw1': _floatarray_feature(np.float32(Tptv).flatten().tolist()),
                            'dns_raw1': _floatarray_feature(np.float32(Tdns).flatten().tolist()),
                            'cbc_raw1': _floatarray_feature(np.float32(Tcbc).flatten().tolist()),
                            'gap_raw1': _floatarray_feature(np.float32(Tgap).flatten().tolist()),
                            'ptv_flag': _floatarray_feature(np.float32(Flag).flatten().tolist()),
                        }
                    )
                )   

            # Write sample in the tfrecord file

            writer.write(example.SerializeToString())

            # Advance the major loop one step

            idx += 1

        # Close Tensorflow writer object
        
        writer.close()

    return

def generate_tfrecords_training(root_folder, us, n_samples_train, max_samples_per_tf, channels, noise):
    """
        Function to generate tfrecords files for the training data.
    
    :param root_folder:        String indicating the folder where the data is stored.
    :param us:                 Integer containing the upsampling ratio between the low- and high-resolution data.
    :param n_samples_train:    Integer indicating the number of samples in the training dataset.
    :param max_samples_per_tf: Integer indicating the maximum number of samples that can be stored in a single tfrecord file.
    """
    

    """
        Store tfrecord files
    """

    # Define path where the tfrecord files will be stored

    tfr_path = f"{root_folder}ss{us:02}/tfrecords/train/"

    # Check if the tfrecord folder exists

    if not os.path.exists(tfr_path):

        # If not, create it

        os.makedirs(tfr_path)

    # Define base for the tfrecords filename

    tfrecords_filename_base = f"{tfr_path}us{us}_noise_{noise}_"

    # Calculate number of tfrecord files that will be generated

    n_sets = int(np.ceil(n_samples_train / max_samples_per_tf))

    # Define index indicating where the training sample should be acquired from

    idx = 0

    # Iterate over the calculated number of tfrecord files

    for n_set in tqdm(range(1, n_sets + 1)):

        # Check if this is the last tfrecord file

        if n_set * max_samples_per_tf < n_samples_train:

            # If not, store the maximum number of samples per tfrecord file

            num_smp = max_samples_per_tf

        else:

            # If this is the last tfrecord file store the remaining samples, always equal or smaller than the maximum number of samples per tfrecord file

            num_smp = n_samples_train - (n_set - 1) * max_samples_per_tf

        # Define filename for the actual tfrecord file

        tfrecords_filename = tfrecords_filename_base + f"file_{n_set:03d}-of-{n_sets:03d}_samples_{num_smp}.tfrecords"

        # Open Tensorflow writer object

        writer = tf.io.TFRecordWriter(tfrecords_filename)

        # Iterate over the number of samples to be stored in the actual tfrecord file

        for i in range(num_smp):

            if channels == 2:
            
                piv_path = f"{root_folder}ss01/piv_noise{noise}/SS1_{(idx+1):06d}.mat"
                ptv_path = f"{root_folder}ss{us:02d}/ptv_noise{noise}/SS{us}_{(idx+1):06d}.mat"

                piv = h5py.File(piv_path, 'r')
                Upiv = np.array(piv['Uptv']).T
                Vpiv = np.array(piv['Vptv']).T

                ptv = h5py.File(ptv_path, 'r')
                Uptv = np.array(ptv['Uptv']).T
                Vptv = np.array(ptv['Vptv']).T
                Flag = np.array(ptv['Flagptv']).T
    
                if idx == 0:

                    # If so, allocate variable for the grid resolution in the high- and low-resolution data

                    nx_piv = Upiv.shape[1]
                    ny_piv = Upiv.shape[0]

                    nx_ptv = Uptv.shape[1]
                    ny_ptv = Uptv.shape[0]

                    grid_path = f"{root_folder}ss01/piv_noise{noise}/SS1_grid.mat"
                    grid = sio.loadmat(grid_path)

                    xlr = np.array(grid['X']).T
                    ylr = np.array(grid['Y']).T
                    
                    grid_path = f"{root_folder}ss{us:02d}/ptv_noise{noise}/SS{us}_grid.mat"
                    grid = sio.loadmat(grid_path)

                    xhr = np.array(grid['X'])
                    yhr = np.array(grid['Y'])

                # Define sample to be stored

                example = tf.train.Example(
                    features = tf.train.Features(
                        feature={
                            'i_sample': _int64_feature(idx),
                            'nx_piv': _int64_feature(int(nx_piv)),
                            'ny_piv': _int64_feature(int(ny_piv)),
                            'nx_ptv': _int64_feature(int(nx_ptv)),
                            'ny_ptv': _int64_feature(int(ny_ptv)),
                            'x_lr': _floatarray_feature(np.float32(xlr).flatten().tolist()),
                            'y_lr': _floatarray_feature(np.float32(ylr).flatten().tolist()),
                            'x_hr': _floatarray_feature(np.float32(xhr).flatten().tolist()),
                            'y_hr': _floatarray_feature(np.float32(yhr).flatten().tolist()),
                            'piv_raw1': _floatarray_feature(np.float32(Upiv).flatten().tolist()),
                            'piv_raw2': _floatarray_feature(np.float32(Vpiv).flatten().tolist()),
                            'ptv_raw1': _floatarray_feature(np.float32(Uptv).flatten().tolist()),
                            'ptv_raw2': _floatarray_feature(np.float32(Vptv).flatten().tolist()),
                            'ptv_flag': _floatarray_feature(np.float32(Flag).flatten().tolist()),
                        }
                    )
                )  

            elif channels==1:

                try: 

                    piv_path = f"{root_folder}ss01/piv_noise{noise}/SS1_{(idx+1):06d}.mat"
                    ptv_path = f"{root_folder}ss{us:02d}/ptv_noise{noise}/SS{us}_{(idx+1):06d}.mat"

                    piv = h5py.File(piv_path, 'r')
                    Tpiv = np.array(piv['T']).T

                    ptv = h5py.File(ptv_path, 'r')
                    Tptv = np.array(ptv['T']).T
                    Flag = np.array(ptv['Flagptv']).T

                except OSError:
                    
                    pass
    
                if idx == 0:

                    # If so, allocate variable for the grid resolution in the high- and low-resolution data

                    nx_piv = Tpiv.shape[1]
                    ny_piv = Tpiv.shape[0]

                    nx_ptv = Tptv.shape[1]
                    ny_ptv = Tptv.shape[0]

                    grid_path = f"{root_folder}ss01/piv_noise{noise}/SS1_grid.mat"
                    grid = sio.loadmat(grid_path)

                    xlr = np.array(grid['X']).T
                    ylr = np.array(grid['Y']).T
                    
                    grid_path = f"{root_folder}ss{us:02d}/ptv_noise{noise}/SS{us}_grid.mat"
                    grid = sio.loadmat(grid_path)

                    xhr = np.array(grid['X'])
                    yhr = np.array(grid['Y'])

                # Define sample to be stored

                example = tf.train.Example(
                    features = tf.train.Features(
                        feature={
                            'i_sample': _int64_feature(idx),
                            'nx_piv': _int64_feature(int(nx_piv)),
                            'ny_piv': _int64_feature(int(ny_piv)),
                            'nx_ptv': _int64_feature(int(nx_ptv)),
                            'ny_ptv': _int64_feature(int(ny_ptv)),
                            'x_lr': _floatarray_feature(np.float32(xlr).flatten().tolist()),
                            'y_lr': _floatarray_feature(np.float32(ylr).flatten().tolist()),
                            'x_hr': _floatarray_feature(np.float32(xhr).flatten().tolist()),
                            'y_hr': _floatarray_feature(np.float32(yhr).flatten().tolist()),
                            'piv_raw1': _floatarray_feature(np.float32(Tpiv).flatten().tolist()),
                            'ptv_raw1': _floatarray_feature(np.float32(Tptv).flatten().tolist()),
                            'ptv_flag': _floatarray_feature(np.float32(Flag).flatten().tolist()),
                        }
                    )
                )  

            # Write sample in the tfrecord file

            writer.write(example.SerializeToString())

            # Advance the major loop one step

            idx += 1

        # Close Tensorflow writer object
        
        writer.close()

    return


@tf.function
def tf_parser(rec, root_folder, us, channels, noise):
    """
        Function to parse the information contained in the tfrecord files.

    :param rec:         Binary file containing the information for one sample.
    :param root_folder: String containg the folder where data is stored
    :return piv:        Tensorflow array with the Particle Image Velocimetry information (wall-normal x streamwise x channels)
    :return ptv:        Tensorflow array with the Particle Tracking Velocimetry information (wall-normal x streamwise x channels)
    :return dns:        Tensorflow array with the Direct Numerical Simulation information (wall-normal x streamwise x channels)
    :return flag:       Tensorflow array with the number of PTV grid point containing information (wall-normal x streamwise x channels)
    """

    """
        Read data
    """

    if channels == 2:
    
        # Define dictionary for the variables' name and type contained in the binary files 

        features = {
            'i_sample': tf.io.FixedLenFeature([], tf.int64),                               # Sample number in the DNS extraction procedure
            'nx_piv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the streamwise direction for the low-resolution data 
            'ny_piv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the wall-normal direction for the low-resolution data 
            'nx_ptv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the streamwise direction for the high-resolution data 
            'ny_ptv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the wall-normal direction for the high-resolution data 
            'x_lr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # Low-resolution domain length in the streamwise direction
            'y_lr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # Low-resolution domain length in the wall-normal direction
            'x_hr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # High-resolution domain length in the streamwise direction
            'y_hr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # High-resolution domain length in the wall-normal direction
            'piv_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the low-resolution data
            'piv_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Wall-normal velocity in the low-resolution data
            'ptv_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the high-resolution data
            'ptv_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Wall-normal velocity in the high-resolution data
            'ptv_flag': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Bins in the high-resolution data containing information
        }

        # Parse binary file

        parsed_rec = tf.io.parse_single_example(rec, features)

        # Read sample number in the DNS extraction procedure

        i_smp = tf.cast(parsed_rec['i_sample'], tf.int32)

        # Read streamwise and wall-normal grid points inthe low-resolution data

        nx_piv = tf.cast(parsed_rec['nx_piv'], tf.int32)
        ny_piv = tf.cast(parsed_rec['ny_piv'], tf.int32)

        # Read streamwise and wall-normal grid points inthe high-resolution data

        nx_ptv = tf.cast(parsed_rec['nx_ptv'], tf.int32)
        ny_ptv = tf.cast(parsed_rec['ny_ptv'], tf.int32)

        """
            Scale data
        """

        # Define path to file containing scaling value

        filename = f"{root_folder}tfrecords/scaling_us{us}_noise_{noise}.npz"

        # Load mean velocity values in the streamwise and wall-normal directions for low- and high-resolution data

        Upiv_mean = np.expand_dims(np.load(filename)['Upiv_mean'], axis=2)
        Vpiv_mean = np.expand_dims(np.load(filename)['Vpiv_mean'], axis=2)
        Uptv_mean = np.expand_dims(np.load(filename)['Uptv_mean'], axis=2)
        Vptv_mean = np.expand_dims(np.load(filename)['Vptv_mean'], axis=2)

        # Load standard deviation velocity values in the streamwise and wall-normal directions for low- and high-resolution data

        Upiv_std = np.expand_dims(np.load(filename)['Upiv_std'], axis=2)
        Vpiv_std = np.expand_dims(np.load(filename)['Vpiv_std'], axis=2)
        Uptv_std = np.expand_dims(np.load(filename)['Uptv_std'], axis=2)
        Vptv_std = np.expand_dims(np.load(filename)['Vptv_std'], axis=2)

        # Reshape flag data into 2-dimensional matrix 

        flag = tf.reshape(parsed_rec['ptv_flag'], (ny_ptv, nx_ptv, 1))

        # Reshape data into 2-dimensional matrix, substract mean value and divide by the standard deviation. Concatenate the streamwise and wall-normal velocities along the third dimension

        piv = (tf.reshape(parsed_rec['piv_raw1'], (ny_piv, nx_piv, 1)) - Upiv_mean) / Upiv_std
        piv = tf.concat((piv, (tf.reshape(parsed_rec['piv_raw2'], (ny_piv, nx_piv, 1)) - Vpiv_mean) / Vpiv_std), -1)
        
        ptv = (tf.reshape(parsed_rec['ptv_raw1'], (ny_ptv, nx_ptv, 1)) - Uptv_mean * flag) / Uptv_std
        ptv = tf.concat((ptv, (tf.reshape(parsed_rec['ptv_raw2'], (ny_ptv, nx_ptv, 1)) - Vptv_mean * flag) / Vptv_std), -1)
        ptv = tf.where(tf.math.is_nan(ptv), tf.zeros_like(ptv), ptv)

    elif channels == 1:

            # Define dictionary for the variables' name and type contained in the binary files 

        features = {
            'i_sample': tf.io.FixedLenFeature([], tf.int64),                               # Sample number in the DNS extraction procedure
            'nx_piv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the streamwise direction for the low-resolution data 
            'ny_piv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the wall-normal direction for the low-resolution data 
            'nx_ptv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the streamwise direction for the high-resolution data 
            'ny_ptv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the wall-normal direction for the high-resolution data 
            'x_lr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # Low-resolution domain length in the streamwise direction
            'y_lr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # Low-resolution domain length in the wall-normal direction
            'x_hr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # High-resolution domain length in the streamwise direction
            'y_hr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # High-resolution domain length in the wall-normal direction
            'piv_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the low-resolution data
            'ptv_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the high-resolution data
            'ptv_flag': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Bins in the high-resolution data containing information
        }

        # Parse binary file

        parsed_rec = tf.io.parse_single_example(rec, features)

        # Read sample number in the DNS extraction procedure

        i_smp = tf.cast(parsed_rec['i_sample'], tf.int32)

        # Read streamwise and wall-normal grid points inthe low-resolution data

        nx_piv = tf.cast(parsed_rec['nx_piv'], tf.int32)
        ny_piv = tf.cast(parsed_rec['ny_piv'], tf.int32)

        # Read streamwise and wall-normal grid points inthe high-resolution data

        nx_ptv = tf.cast(parsed_rec['nx_ptv'], tf.int32)
        ny_ptv = tf.cast(parsed_rec['ny_ptv'], tf.int32)

        """
            Scale data
        """

        # Define path to file containing scaling value

        filename = f"{root_folder}tfrecords/scaling_us{us}_noise_{noise}.npz"

        # Load mean velocity values in the streamwise and wall-normal directions for low- and high-resolution data

        Tpiv_mean = np.expand_dims(np.load(filename)['Tpiv_mean'], axis=2)
        Tptv_mean = np.expand_dims(np.load(filename)['Tptv_mean'], axis=2)

        # Load standard deviation velocity values in the streamwise and wall-normal directions for low- and high-resolution data

        Tpiv_std = np.expand_dims(np.load(filename)['Tpiv_std'], axis=2)
        Tptv_std = np.expand_dims(np.load(filename)['Tptv_std'], axis=2)

        # Reshape flag data into 2-dimensional matrix 

        flag = tf.reshape(parsed_rec['ptv_flag'], (ny_ptv, nx_ptv, 1))

        # Reshape data into 2-dimensional matrix, substract mean value and divide by the standard deviation. Concatenate the streamwise and wall-normal velocities along the third dimension

        piv = (tf.reshape(parsed_rec['piv_raw1'], (ny_piv, nx_piv, 1)) - Tpiv_mean) / Tpiv_std
        
        ptv = (tf.reshape(parsed_rec['ptv_raw1'], (ny_ptv, nx_ptv, 1)) - Tptv_mean * flag) / Tptv_std
        piv = tf.where(tf.math.is_nan(piv), tf.zeros_like(piv), piv)
        ptv = tf.where(tf.math.is_nan(ptv), tf.zeros_like(ptv), ptv)

    return piv, ptv, flag


@tf.function
def tf_parser_testing(rec, root_folder, us, channels, noise):
    """
        Function to parse the information contained in the tfrecord files.

    :param rec:         Binary file containing the information for one sample.
    :param root_folder: String containg the folder where data is stored
    :return piv:        Tensorflow array with the Particle Image Velocimetry information (wall-normal x streamwise x channels)
    :return ptv:        Tensorflow array with the Particle Tracking Velocimetry information (wall-normal x streamwise x channels)
    :return dns:        Tensorflow array with the Direct Numerical Simulation information (wall-normal x streamwise x channels)
    :return flag:       Tensorflow array with the number of PTV grid point containing information (wall-normal x streamwise x channels)
    """

    """
        Read data
    """
    
    # Define dictionary for the variables' name and type contained in the binary files 

    if channels == 2:

        features = {
            'i_sample': tf.io.FixedLenFeature([], tf.int64),                               # Sample number in the DNS extraction procedure
            'nx_piv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the streamwise direction for the low-resolution data 
            'ny_piv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the wall-normal direction for the low-resolution data 
            'nx_ptv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the streamwise direction for the high-resolution data 
            'ny_ptv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the wall-normal direction for the high-resolution data 
            'x_lr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # Low-resolution domain length in the streamwise direction
            'y_lr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # Low-resolution domain length in the wall-normal direction
            'x_hr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # High-resolution domain length in the streamwise direction
            'y_hr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # High-resolution domain length in the wall-normal direction
            'piv_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the low-resolution data
            'piv_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Wall-normal velocity in the low-resolution data
            'ptv_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the high-resolution data
            'ptv_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Wall-normal velocity in the high-resolution data
            'dns_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the original data
            'dns_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Wall-normal velocity in the original data
            'cbc_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the original data
            'cbc_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Wall-normal velocity in the original data
            'gap_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the original data
            'gap_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Wall-normal velocity in the original data
            'ptv_flag': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Bins in the high-resolution data containing information
        }

        # Parse binary file

        parsed_rec = tf.io.parse_single_example(rec, features)

        # Read sample number in the DNS extraction procedure

        i_smp = tf.cast(parsed_rec['i_sample'], tf.int32)

        # Read streamwise and wall-normal grid points inthe low-resolution data

        nx_piv = tf.cast(parsed_rec['nx_piv'], tf.int32)
        ny_piv = tf.cast(parsed_rec['ny_piv'], tf.int32)

        # Read streamwise and wall-normal grid points inthe high-resolution data

        nx_ptv = tf.cast(parsed_rec['nx_ptv'], tf.int32)
        ny_ptv = tf.cast(parsed_rec['ny_ptv'], tf.int32)

        """
            Scale data
        """

        # Define path to file containing scaling value

        filename = f"{root_folder}tfrecords/scaling_us{us}_noise_{noise}.npz"

        # Load mean velocity values in the streamwise and wall-normal directions for low- and high-resolution data

        Upiv_mean = np.expand_dims(np.load(filename)['Upiv_mean'], axis=2)
        Vpiv_mean = np.expand_dims(np.load(filename)['Vpiv_mean'], axis=2)
        Uptv_mean = np.expand_dims(np.load(filename)['Uptv_mean'], axis=2)
        Vptv_mean = np.expand_dims(np.load(filename)['Vptv_mean'], axis=2)

        # Load standard deviation velocity values in the streamwise and wall-normal directions for low- and high-resolution data

        Upiv_std = np.expand_dims(np.load(filename)['Upiv_std'], axis=2)
        Vpiv_std = np.expand_dims(np.load(filename)['Vpiv_std'], axis=2)
        Uptv_std = np.expand_dims(np.load(filename)['Uptv_std'], axis=2)
        Vptv_std = np.expand_dims(np.load(filename)['Vptv_std'], axis=2)

        # Reshape flag data into 2-dimensional matrix 

        flag = tf.reshape(parsed_rec['ptv_flag'], (ny_ptv, nx_ptv, 1))
        

        # Reshape data into 2-dimensional matrix, substract mean value and divide by the standard deviation. Concatenate the streamwise and wall-normal velocities along the third dimension


        piv = (tf.reshape(parsed_rec['piv_raw1'], (ny_piv, nx_piv, 1)) - Upiv_mean) / Upiv_std
        piv = tf.concat((piv, (tf.reshape(parsed_rec['piv_raw2'], (ny_piv, nx_piv, 1)) - Vpiv_mean) / Vpiv_std), -1)
            
        ptv = (tf.reshape(parsed_rec['ptv_raw1'], (ny_ptv, nx_ptv, 1)) - Uptv_mean * flag) / Uptv_std
        ptv = tf.concat((ptv, (tf.reshape(parsed_rec['ptv_raw2'], (ny_ptv, nx_ptv, 1)) - Vptv_mean * flag) / Vptv_std), -1)

        dns = tf.reshape(parsed_rec['dns_raw1'], (ny_ptv, nx_ptv, 1))
        dns = tf.concat((dns, tf.reshape(parsed_rec['dns_raw2'], (ny_ptv, nx_ptv, 1))), -1)

        cbc = tf.reshape(parsed_rec['cbc_raw1'], (ny_ptv, nx_ptv, 1))
        cbc = tf.concat((cbc, tf.reshape(parsed_rec['cbc_raw2'], (ny_ptv, nx_ptv, 1))), -1)
        gap = tf.reshape(parsed_rec['gap_raw1'], (ny_ptv, nx_ptv, 1))
        gap = tf.concat((gap, tf.reshape(parsed_rec['gap_raw2'], (ny_ptv, nx_ptv, 1))), -1)

    elif channels == 1:

        features = {
            'i_sample': tf.io.FixedLenFeature([], tf.int64),                               # Sample number in the DNS extraction procedure
            'nx_piv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the streamwise direction for the low-resolution data 
            'ny_piv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the wall-normal direction for the low-resolution data 
            'nx_ptv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the streamwise direction for the high-resolution data 
            'ny_ptv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the wall-normal direction for the high-resolution data 
            'x_lr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # Low-resolution domain length in the streamwise direction
            'y_lr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # Low-resolution domain length in the wall-normal direction
            'x_hr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # High-resolution domain length in the streamwise direction
            'y_hr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # High-resolution domain length in the wall-normal direction
            'piv_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the low-resolution data
            'ptv_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the high-resolution data
            'dns_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the original data
            'cbc_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the original data
            'gap_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'ptv_flag': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Bins in the high-resolution data containing information
        }

        # Parse binary file

        parsed_rec = tf.io.parse_single_example(rec, features)

        # Read sample number in the DNS extraction procedure

        i_smp = tf.cast(parsed_rec['i_sample'], tf.int32)

        # Read streamwise and wall-normal grid points inthe low-resolution data

        nx_piv = tf.cast(parsed_rec['nx_piv'], tf.int32)
        ny_piv = tf.cast(parsed_rec['ny_piv'], tf.int32)

        # Read streamwise and wall-normal grid points inthe high-resolution data

        nx_ptv = tf.cast(parsed_rec['nx_ptv'], tf.int32)
        ny_ptv = tf.cast(parsed_rec['ny_ptv'], tf.int32)

        """
            Scale data
        """

        # Define path to file containing scaling value

        filename = f"{root_folder}tfrecords/scaling_us{us}_noise_{noise}.npz"

        # Load mean velocity values in the streamwise and wall-normal directions for low- and high-resolution data

        Tpiv_mean = np.expand_dims(np.load(filename)['Tpiv_mean'], axis=2)
        Tptv_mean = np.expand_dims(np.load(filename)['Tptv_mean'], axis=2)

        # Load standard deviation velocity values in the streamwise and wall-normal directions for low- and high-resolution data

        Tpiv_std = np.expand_dims(np.load(filename)['Tpiv_std'], axis=2)
        Tptv_std = np.expand_dims(np.load(filename)['Tptv_std'], axis=2)

        # Reshape flag data into 2-dimensional matrix 

        flag = tf.reshape(parsed_rec['ptv_flag'], (ny_ptv, nx_ptv, 1))

        # Reshape data into 2-dimensional matrix, substract mean value and divide by the standard deviation. Concatenate the streamwise and wall-normal velocities along the third dimension

        piv = (tf.reshape(parsed_rec['piv_raw1'], (ny_piv, nx_piv, 1)) - Tpiv_mean) / Tpiv_std
        ptv = (tf.reshape(parsed_rec['ptv_raw1'], (ny_ptv, nx_ptv, 1)) - Tptv_mean * flag) / Tptv_std
        piv = tf.where(tf.math.is_nan(piv), tf.zeros_like(piv), piv)
        ptv = tf.where(tf.math.is_nan(ptv), tf.zeros_like(ptv), ptv)
        dns = tf.reshape(parsed_rec['dns_raw1'], (ny_ptv, nx_ptv, 1))
        cbc = tf.reshape(parsed_rec['cbc_raw1'], (ny_ptv, nx_ptv, 1))
        gap = tf.reshape(parsed_rec['gap_raw1'], (ny_ptv, nx_ptv, 1))
    
    xlr = tf.reshape(parsed_rec['x_lr'], (ny_piv, nx_piv))
    ylr = tf.reshape(parsed_rec['y_lr'], (ny_piv, nx_piv))
    xhr = tf.reshape(parsed_rec['x_hr'], (ny_ptv, nx_ptv))
    yhr = tf.reshape(parsed_rec['y_hr'], (ny_ptv, nx_ptv))

    return piv, ptv, dns, cbc, gap, flag, xlr, ylr, xhr, yhr


def _floatarray_feature(value):

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


<<<<<<< HEAD
=======


















"""
    OLD
"""

def generate_pipeline_timeresolved(root_folder, time_folder):
    """
        Function to generate Tensorflow data pipeline for the testing dataset.

    :param root_folder:         String indicating the folder where the data is stored.
    :return dataset_test:       Tensorflow object containing the pipeline for the testing dataset.
    :return tot_samples_per_ds: Integer showing the total number of samples in the dataset
    """

    """
        Prepare files
    """

    # Define path where the tfrecord files are stored

    tfr_path = f"{time_folder}tfrecords/"

    # Select files in the tfrecord folder

    tfr_files = sorted([os.path.join(tfr_path,f) for f in os.listdir(tfr_path) if os.path.isfile(os.path.join(tfr_path,f))])

    # Keep only files terminated on .tfrecords

    regex = re.compile(f'.tfrecords')

    tfr_files = ([string for string in tfr_files if re.search(regex, string)])

    # Find number of samples for each tfrecord file
    
    n_samples_per_tfr = np.array([int(s.split('.')[-2][-2:].replace('_', '')) for s in tfr_files])
    n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]

    # Define total number of samples in the tfrecord files

    tot_samples_per_ds = sum(n_samples_per_tfr)

    # Initialize the testing dataset with the assigned tfrecord files. Internal shuffle is not applied

    tfr_files_test_ds = tf.data.Dataset.list_files(tfr_files, shuffle=False)

    # Parse the tfrecord files assigned to the testing dataset

    tfr_files_test_ds = tf.data.TFRecordDataset(tfr_files_test_ds)

    """
        Generated testing dataset pipeline
    """

    # Parse the information contained in the tfrecord files

    dataset_test = tfr_files_test_ds.map(lambda x: tf_parser(x, root_folder), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset_test, tot_samples_per_ds

def generate_scaling_data_exptbl(root_folder, nx, ny, us, n_samples_train):
    """
        Function to generate the scaling values from the training dataset.
    
    :param root_folder:        String indicating the folder where the data is stored.
    :param nx:                 Integer containing the grid points in the streamwise direction for the low-resolution data.
    :param ny:                 Integer containing the grid points in the wall-normal direction for the low-resolution data.
    :param us:                 Integer containing the upsampling ratio between the low- and high-resolution data.
    :param n_samples_train:    Integer indicating the number of samples in the training dataset.
    """


    Upiv = np.zeros((n_samples_train, ny,    nx))
    Vpiv = np.zeros((n_samples_train, ny,    nx))
    Uptv = np.zeros((n_samples_train, ny*us, nx*us))
    Vptv = np.zeros((n_samples_train, ny*us, nx*us))
    Flag = np.zeros((n_samples_train, ny*us, nx*us))

    # Iterate over the number of samples

    for idx in tqdm(range(n_samples_train)):

        # Define path to Matlab file containing the actual sample

        piv_path = f"{root_folder}ss01/piv/PIV_{(idx+1):06d}.mat"
        ptv_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_{(idx+1):06d}.mat"
            
        # Load Matlab file

        piv = h5py.File(piv_path, 'r')
        ptv = h5py.File(ptv_path, 'r')

        # Write each variable in their corresponding global matrices

        Upiv[idx, :, :] = np.array(piv['U']).T[:, 1:]
        Vpiv[idx, :, :] = np.array(piv['V']).T[:, 1:]
        Uptv[idx, :, :] = np.array(ptv['Uptv']).T[:, us:]
        Vptv[idx, :, :] = np.array(ptv['Vptv']).T[:, us:]
        Flag[idx, :, :] = np.array(ptv['Flagptv']).T[:, us:]

    # Compute mean values

    Upiv_mean = np.mean(Upiv, axis=0)
    Vpiv_mean = np.mean(Vpiv, axis=0)
    Uptv_mean = np.sum(Uptv, axis=0) / np.sum(Flag, axis=0)
    Vptv_mean = np.sum(Vptv, axis=0) / np.sum(Flag, axis=0)

    Upiv_mean = np.nanmean(Upiv_mean, axis=1, keepdims=True)
    Vpiv_mean = np.nanmean(Vpiv_mean, axis=1, keepdims=True)
    Uptv_mean = np.nanmean(Uptv_mean, axis=1, keepdims=True)
    Vptv_mean = np.nanmean(Vptv_mean, axis=1, keepdims=True)

    # Compute standard deviation values
    
    Upiv_std = np.std(Upiv, axis=0)
    Vpiv_std = np.std(Vpiv, axis=0)

    del Upiv, Vpiv, Uptv, Vptv
    
    Uptv_std = np.zeros((ny*us, nx*us))
    Vptv_std = np.zeros((ny*us, nx*us))

    for idx in tqdm(range(n_samples_train)):

        # Define path to Matlab file containing the actual sample

        ptv_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_{(idx+1):06d}.mat"
            
        # Load Matlab file

        ptv = h5py.File(ptv_path, 'r')

        # Write each variable in their corresponding global matrices


        Uptv_std += (np.array(ptv['Uptv']).T[:, us:] - Uptv_mean * np.array(ptv['Flagptv']).T[:, us:]) ** 2
        Vptv_std += (np.array(ptv['Vptv']).T[:, us:] - Vptv_mean * np.array(ptv['Flagptv']).T[:, us:]) ** 2

    Uptv_std = np.sqrt(Uptv_std / np.sum(Flag, axis=0))
    Vptv_std = np.sqrt(Vptv_std / np.sum(Flag, axis=0))

    Upiv_std = np.nanmean(Upiv_std, axis=1, keepdims=True)
    Vpiv_std = np.nanmean(Vpiv_std, axis=1, keepdims=True)
    Uptv_std = np.nanmean(Uptv_std, axis=1, keepdims=True)
    Vptv_std = np.nanmean(Vptv_std, axis=1, keepdims=True)
    
    # Define path to stored the scaling values

    filename = f"{root_folder}ss{us:02}/tfrecords/scaling.npz"

    # Save scaling values

    np.savez(
        filename,
        Upiv_mean=Upiv_mean,
        Vpiv_mean=Vpiv_mean,
        Uptv_mean=Uptv_mean,
        Vptv_mean=Vptv_mean,
        Upiv_std=Upiv_std,
        Vpiv_std=Vpiv_std,
        Uptv_std=Uptv_std,
        Vptv_std=Vptv_std,
    )

    return



def generate_tfrecords_testing_exptbl(root_folder, us, n_samples_train, n_samples_test, max_samples_per_tf):
    """
        Function to generate tfrecords files for the training data.
    
    :param root_folder:        String indicating the folder where the data is stored.
    :param us:                 Integer containing the upsampling ratio between the low- and high-resolution data.
    :param n_samples_train:    Integer indicating the number of samples in the training dataset.
    :param max_samples_per_tf: Integer indicating the maximum number of samples that can be stored in a single tfrecord file.
    """
    
    bin_size = int(64/us)

    """
        Store tfrecord files
    """

    # Define path where the tfrecord files will be stored

    tfr_path = f"{root_folder}ss{us:02}/tfrecords/test/"

    # Check if the tfrecord folder exists

    if not os.path.exists(tfr_path):

        # If not, create it

        os.makedirs(tfr_path)

    # Define base for the tfrecords filename

    tfrecords_filename_base = f"{tfr_path}Nppp001_bin{bin_size}_ss{us}_PIVw64_PIVdx16_"

    # Calculate number of tfrecord files that will be generated

    n_sets = int(np.ceil(n_samples_test / max_samples_per_tf))

    # Define index indicating where the training sample should be acquired from

    idx = n_samples_train

    # Iterate over the calculated number of tfrecord files

    for n_set in tqdm(range(1, n_sets + 1)):

        # Check if this is the last tfrecord file

        if n_set * max_samples_per_tf < n_samples_test:

            # If not, store the maximum number of samples per tfrecord file

            num_smp = max_samples_per_tf

        else:

            # If this is the last tfrecord file store the remaining samples, always equal or smaller than the maximum number of samples per tfrecord file

            num_smp = n_samples_test - (n_set - 1) * max_samples_per_tf

        # Define filename for the actual tfrecord file

        tfrecords_filename = tfrecords_filename_base + f"file_{n_set:03d}-of-{n_sets:03d}_samples_{num_smp}.tfrecords"

        # Open Tensorflow writer object

        writer = tf.io.TFRecordWriter(tfrecords_filename)

        # Iterate over the number of samples to be stored in the actual tfrecord file

        for i in range(num_smp):

            # Check if this is the first sample
            
            piv_path = f"{root_folder}ss01/piv/PIV_{(idx+1):06d}.mat"
            ptv_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_{(idx+1):06d}.mat"
            dns_path = f"{root_folder}PIV_HR/PIV_HR_{(idx+1):06d}.mat"
            
            # Load Matlab file

            piv = h5py.File(piv_path, 'r')
        
            Upiv = np.array(piv['U']).T[:,1:]
            Vpiv = np.array(piv['V']).T[:,1:]
            
            ptv = h5py.File(ptv_path, 'r')
            
            # Generate matrices for each component
        
            Uptv = np.array(ptv['Uptv']).T[:, us:]
            Vptv = np.array(ptv['Vptv']).T[:, us:]
            Flag = np.array(ptv['Flagptv']).T[:, us:]

            dns = h5py.File(dns_path, 'r')
            Udns = np.array(dns['U']).T
            Vdns = np.array(dns['V']).T

            if us == 2:
                dns = h5py.File(dns_path, 'r')
                Udns = np.array(dns['U']).T
                Vdns = np.array(dns['V']).T
                Udns = Udns[1::2, 4::2]
                Vdns = Vdns[1::2, 4::2]

            else:

                dns = h5py.File(dns_path, 'r')
                Udns = np.array(dns['U']).T[:, us:]
                Vdns = np.array(dns['V']).T[:, us:]


            if idx == n_samples_train:

                # If so, allocate variable for the grid resolution in the high- and low-resolution data

                nx_piv = Upiv.shape[1]
                ny_piv = Upiv.shape[0]

                nx_ptv = Uptv.shape[1]
                ny_ptv = Uptv.shape[0]

                xlr = np.array(piv['XPIV']).T[:,1:]
                ylr = np.array(piv['YPIV']).T[:,1:]

                grid_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_grid.mat"

                grid = sio.loadmat(grid_path)

                xhr = grid['X'][:, us:]
                yhr = grid['Y'][:, us:]
            
            # Define sample to be stored

            example = tf.train.Example(
                features = tf.train.Features(
                    feature={
                        'i_sample': _int64_feature(idx),
                        'nx_piv': _int64_feature(int(nx_piv)),
                        'ny_piv': _int64_feature(int(ny_piv)),
                        'nx_ptv': _int64_feature(int(nx_ptv)),
                        'ny_ptv': _int64_feature(int(ny_ptv)),
                        'x_lr': _floatarray_feature(np.float32(xlr).flatten().tolist()),
                        'y_lr': _floatarray_feature(np.float32(ylr).flatten().tolist()),
                        'x_hr': _floatarray_feature(np.float32(xhr).flatten().tolist()),
                        'y_hr': _floatarray_feature(np.float32(yhr).flatten().tolist()),
                        'piv_raw1': _floatarray_feature(np.float32(Upiv).flatten().tolist()),
                        'piv_raw2': _floatarray_feature(np.float32(Vpiv).flatten().tolist()),
                        'ptv_raw1': _floatarray_feature(np.float32(Uptv).flatten().tolist()),
                        'ptv_raw2': _floatarray_feature(np.float32(Vptv).flatten().tolist()),
                        'dns_raw1': _floatarray_feature(np.float32(Udns).flatten().tolist()),
                        'dns_raw2': _floatarray_feature(np.float32(Vdns).flatten().tolist()),
                        'ptv_flag': _floatarray_feature(np.float32(Flag).flatten().tolist()),
                    }
                )
            )  

            # Write sample in the tfrecord file

            writer.write(example.SerializeToString())

            # Advance the major loop one step

            idx += 1

        # Close Tensorflow writer object
        
        writer.close()

    return


def generate_channel_tfrecords_training(root_folder, us, n_samples_train, max_samples_per_tf):
    """
        Function to generate tfrecords files for the training data.
    
    :param root_folder:        String indicating the folder where the data is stored.
    :param us:                 Integer containing the upsampling ratio between the low- and high-resolution data.
    :param n_samples_train:    Integer indicating the number of samples in the training dataset.
    :param max_samples_per_tf: Integer indicating the maximum number of samples that can be stored in a single tfrecord file.
    """
    
    bin_size = int(32/us)

    """
        Store tfrecord files
    """

    # Define path where the tfrecord files will be stored

    tfr_path = f"{root_folder}ss{us:02}/tfrecords/train/"

    # Check if the tfrecord folder exists

    if not os.path.exists(tfr_path):

        # If not, create it

        os.makedirs(tfr_path)

    # Define base for the tfrecords filename

    tfrecords_filename_base = f"{tfr_path}Nppp001_bin{bin_size}_ss{us}_PIVw32_PIVdx16_"

    # Calculate number of tfrecord files that will be generated

    n_sets = int(np.ceil(n_samples_train / max_samples_per_tf))

    # Define index indicating where the training sample should be acquired from

    idx = 0

    # Iterate over the calculated number of tfrecord files

    piv_path = f"{root_folder}PIV_IW32_GD16.mat"
    piv = sio.loadmat(piv_path)
    Upiv = piv['TPIV'][:, 0, : , :]
    Vpiv = piv['TPIV'][:, 1, : , :]

    for n_set in tqdm(range(1, n_sets + 1)):

        # Check if this is the last tfrecord file

        if n_set * max_samples_per_tf < n_samples_train:

            # If not, store the maximum number of samples per tfrecord file

            num_smp = max_samples_per_tf

        else:

            # If this is the last tfrecord file store the remaining samples, always equal or smaller than the maximum number of samples per tfrecord file

            num_smp = n_samples_train - (n_set - 1) * max_samples_per_tf

        # Define filename for the actual tfrecord file

        tfrecords_filename = tfrecords_filename_base + f"file_{n_set:03d}-of-{n_sets:03d}_samples_{num_smp}.tfrecords"

        # Open Tensorflow writer object

        writer = tf.io.TFRecordWriter(tfrecords_filename)

        # Iterate over the number of samples to be stored in the actual tfrecord file

        for i in range(num_smp):

            # Check if this is the first sample
            
            ptv_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_{(idx+1):06d}.mat"
            
            # Load Matlab file
            
            ptv = h5py.File(ptv_path, 'r')

            # Generate matrices for each component
        
            Uptv = np.array(ptv['Uptv']).T
            Vptv = np.array(ptv['Vptv']).T
            Flag = np.array(ptv['Flagptv']).T
 
            if idx == 0:

                # If so, allocate variable for the grid resolution in the high- and low-resolution data

                nx_piv = Upiv.shape[2]
                ny_piv = Upiv.shape[1]

                nx_ptv = Uptv.shape[1]
                ny_ptv = Uptv.shape[0]

                xlr = np.array(piv['XPIV'])
                ylr = np.array(piv['YPIV'])
                
                xhr = xlr * us
                yhr = ylr * us

            # Define sample to be stored

            example = tf.train.Example(
                features = tf.train.Features(
                    feature={
                        'i_sample': _int64_feature(idx),
                        'nx_piv': _int64_feature(int(nx_piv)),
                        'ny_piv': _int64_feature(int(ny_piv)),
                        'nx_ptv': _int64_feature(int(nx_ptv)),
                        'ny_ptv': _int64_feature(int(ny_ptv)),
                        'x_lr': _floatarray_feature(np.float32(xlr).flatten().tolist()),
                        'y_lr': _floatarray_feature(np.float32(ylr).flatten().tolist()),
                        'x_hr': _floatarray_feature(np.float32(xhr).flatten().tolist()),
                        'y_hr': _floatarray_feature(np.float32(yhr).flatten().tolist()),
                        'piv_raw1': _floatarray_feature(np.float32(Upiv[idx]).flatten().tolist()),
                        'piv_raw2': _floatarray_feature(np.float32(Vpiv[idx]).flatten().tolist()),
                        'ptv_raw1': _floatarray_feature(np.float32(Uptv).flatten().tolist()),
                        'ptv_raw2': _floatarray_feature(np.float32(Vptv).flatten().tolist()),
                        'ptv_flag': _floatarray_feature(np.float32(Flag).flatten().tolist()),
                    }
                )
            )  

            # Write sample in the tfrecord file

            writer.write(example.SerializeToString())

            # Advance the major loop one step

            idx += 1

        # Close Tensorflow writer object
        
        writer.close()

    return


def generate_channel_tfrecords_testing(root_folder, us, n_samples_train, n_samples_test, max_samples_per_tf):
    """
        Function to generate tfrecords files for the training data.
    
    :param root_folder:        String indicating the folder where the data is stored.
    :param us:                 Integer containing the upsampling ratio between the low- and high-resolution data.
    :param n_samples_train:    Integer indicating the number of samples in the training dataset.
    :param max_samples_per_tf: Integer indicating the maximum number of samples that can be stored in a single tfrecord file.
    """
    
    bin_size = int(32/us)

    """
        Store tfrecord files
    """

    # Define path where the tfrecord files will be stored

    tfr_path = f"{root_folder}ss{us:02}/tfrecords/test/"

    # Check if the tfrecord folder exists

    if not os.path.exists(tfr_path):

        # If not, create it

        os.makedirs(tfr_path)

    # Define base for the tfrecords filename

    tfrecords_filename_base = f"{tfr_path}Nppp001_bin{bin_size}_ss{us}_PIVw32_PIVdx16_"

    # Calculate number of tfrecord files that will be generated

    n_sets = int(np.ceil(n_samples_test / max_samples_per_tf))

    # Define index indicating where the training sample should be acquired from

    idx = n_samples_train

    # Iterate over the calculated number of tfrecord files

    piv_path = f"{root_folder}PIV_IW32_GD16.mat"
    piv = sio.loadmat(piv_path)
    Upiv = piv['TPIV'][:, 0, : , :]
    Vpiv = piv['TPIV'][:, 1, : , :]

    for n_set in tqdm(range(1, n_sets + 1)):

        # Check if this is the last tfrecord file

        if n_set * max_samples_per_tf < n_samples_test:

            # If not, store the maximum number of samples per tfrecord file

            num_smp = max_samples_per_tf

        else:

            # If this is the last tfrecord file store the remaining samples, always equal or smaller than the maximum number of samples per tfrecord file

            num_smp = n_samples_test - (n_set - 1) * max_samples_per_tf

        # Define filename for the actual tfrecord file

        tfrecords_filename = tfrecords_filename_base + f"file_{n_set:03d}-of-{n_sets:03d}_samples_{num_smp}.tfrecords"

        # Open Tensorflow writer object

        writer = tf.io.TFRecordWriter(tfrecords_filename)

        # Iterate over the number of samples to be stored in the actual tfrecord file

        for i in range(num_smp):

            # Check if this is the first sample
            
            ptv_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_{(idx+1):06d}.mat"
            dns_path = f"{root_folder}ss{us:02d}/dns/DNS_SS{int(us*2)}_{(idx+1):06d}.mat"
            cbc_path = f"{root_folder}ss{us:02d}/cubic/cubic_SS{us}_{(idx+1):06d}.mat"
            
            # Load Matlab file
            
            ptv = h5py.File(ptv_path, 'r')

            # Generate matrices for each component
        
            Uptv = np.array(ptv['Uptv']).T
            Vptv = np.array(ptv['Vptv']).T
            Flag = np.array(ptv['Flagptv']).T
            
            dns = h5py.File(dns_path, 'r')
            
            # # Generate matrices for each component
        
            Udns = np.array(dns['UDNS']).T
            Vdns = np.array(dns['VDNS']).T

            cbc = sio.loadmat(cbc_path)
            
            # # Generate matrices for each component
        
            Ucbc = np.array(cbc['U_interp'])
            Vcbc = np.array(cbc['V_interp'])

            if idx == n_samples_train:

                # If so, allocate variable for the grid resolution in the high- and low-resolution data

                nx_piv = Upiv.shape[2]
                ny_piv = Upiv.shape[1]

                nx_ptv = Uptv.shape[1]
                ny_ptv = Uptv.shape[0]

                xlr = np.array(piv['XPIV'])
                ylr = np.array(piv['YPIV'])

                grid_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_grid.mat"

                grid = sio.loadmat(grid_path)

                xhr = grid['X']
                yhr = grid['Y']

            # Define sample to be stored

            example = tf.train.Example(
                features = tf.train.Features(
                    feature={
                        'i_sample': _int64_feature(idx),
                        'nx_piv': _int64_feature(int(nx_piv)),
                        'ny_piv': _int64_feature(int(ny_piv)),
                        'nx_ptv': _int64_feature(int(nx_ptv)),
                        'ny_ptv': _int64_feature(int(ny_ptv)),
                        'x_lr': _floatarray_feature(np.float32(xlr).flatten().tolist()),
                        'y_lr': _floatarray_feature(np.float32(ylr).flatten().tolist()),
                        'x_hr': _floatarray_feature(np.float32(xhr).flatten().tolist()),
                        'y_hr': _floatarray_feature(np.float32(yhr).flatten().tolist()),
                        'piv_raw1': _floatarray_feature(np.float32(Upiv[idx]).flatten().tolist()),
                        'piv_raw2': _floatarray_feature(np.float32(Vpiv[idx]).flatten().tolist()),
                        'ptv_raw1': _floatarray_feature(np.float32(Uptv).flatten().tolist()),
                        'ptv_raw2': _floatarray_feature(np.float32(Vptv).flatten().tolist()),
                        'dns_raw1': _floatarray_feature(np.float32(Udns).flatten().tolist()),
                        'dns_raw2': _floatarray_feature(np.float32(Vdns).flatten().tolist()),
                        'cbc_raw1': _floatarray_feature(np.float32(Ucbc).flatten().tolist()),
                        'cbc_raw2': _floatarray_feature(np.float32(Vcbc).flatten().tolist()),
                        'ptv_flag': _floatarray_feature(np.float32(Flag).flatten().tolist()),
                    }
                )
            )  

            # Write sample in the tfrecord file

            writer.write(example.SerializeToString())

            # Advance the major loop one step

            idx += 1

        # Close Tensorflow writer object
        
        writer.close()

    return


def generate_channel_scaling_data(root_folder, nx, ny, us, n_samples_train):
    """
        Function to generate the scaling values from the training dataset.
    
    :param root_folder:        String indicating the folder where the data is stored.
    :param nx:                 Integer containing the grid points in the streamwise direction for the low-resolution data.
    :param ny:                 Integer containing the grid points in the wall-normal direction for the low-resolution data.
    :param us:                 Integer containing the upsampling ratio between the low- and high-resolution data.
    :param n_samples_train:    Integer indicating the number of samples in the training dataset.
    """


    piv_path = f"{root_folder}PIV_IW32_GD16.mat"
    piv = sio.loadmat(piv_path)
    Upiv = piv['TPIV'][:, 0, : , :]
    Vpiv = piv['TPIV'][:, 1, : , :]
    Uptv = np.zeros((n_samples_train, ny*us, nx*us))
    Vptv = np.zeros((n_samples_train, ny*us, nx*us))
    Flag = np.zeros((n_samples_train, ny*us, nx*us))

    # Iterate over the number of samples

    for idx in tqdm(range(n_samples_train)):

        # Define path to Matlab file containing the actual sample

        ptv_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_{(idx+1):06d}.mat"
            
        # Load Matlab file

        ptv = h5py.File(ptv_path, 'r')

        # Write each variable in their corresponding global matrices

        Uptv[idx, :, :] = np.array(ptv['Uptv']).T
        Vptv[idx, :, :] = np.array(ptv['Vptv']).T
        Flag[idx, :, :] = np.array(ptv['Flagptv']).T

    # Compute mean values

    Upiv_mean = np.mean(Upiv, axis=0)
    Vpiv_mean = np.mean(Vpiv, axis=0)
    Uptv_mean = np.sum(Uptv, axis=0) / np.sum(Flag, axis=0)
    Vptv_mean = np.sum(Vptv, axis=0) / np.sum(Flag, axis=0)

    for idx in range(int(nx*us)):

        if np.isnan(Uptv_mean[:, idx]).any():

            continue

        else:

            for idz in range(idx):

                Uptv_mean[:, idz] = Uptv_mean[:, idx]
                Vptv_mean[:, idz] = Uptv_mean[:, idx]

            break

    for idx in range(int(nx*us)):

        for idy in range(int(ny*us)):

            if np.isnan(Uptv_mean[idy, idx]):

                Uptv_mean[idy, idx] = Uptv_mean[idy - 1, idx-1]
                Vptv_mean[idy, idx] = Uptv_mean[idy - 1, idx-1]

    # Compute standard deviation values
    
    Upiv_std = np.std(Upiv, axis=0)
    Vpiv_std = np.std(Vpiv, axis=0)
    Uptv_std = np.sqrt(np.sum((Uptv - Uptv_mean*Flag)**2, axis=0) / np.sum(Flag, axis=0))
    Vptv_std = np.sqrt(np.sum((Vptv - Vptv_mean*Flag)**2, axis=0) / np.sum(Flag, axis=0))
    
    for idx in range(int(nx*us)):

        if np.isnan(Uptv_std[:, idx]).any():

            continue

        else:

            for idz in range(idx):

                Uptv_std[:, idz] = Uptv_std[:, idx]
                Vptv_std[:, idz] = Uptv_std[:, idx]

            break


    for idx in range(int(nx*us)):

        for idy in range(int(ny*us)):

            if np.isnan(Uptv_std[idy, idx]):

                Uptv_std[idy, idx] = Uptv_std[idy - 1, idx-1]
                Vptv_std[idy, idx] = Uptv_std[idy - 1, idx-1]

    for idx in range(int(nx*us)):

        for idy in range(int(ny*us)):

            if Uptv_std[idy, idx] == 0:

                Uptv_std[idy, idx] = Uptv_std[idy - 1, idx-1]
                Vptv_std[idy, idx] = Uptv_std[idy - 1, idx-1]

    # Define path to stored the scaling values

    filename = f"{root_folder}ss{us:02}/tfrecords/scaling.npz"

    # Save scaling values

    np.savez(
        filename,
        Upiv_mean=Upiv_mean,
        Vpiv_mean=Vpiv_mean,
        Uptv_mean=Uptv_mean,
        Vptv_mean=Vptv_mean,
        Upiv_std=Upiv_std,
        Vpiv_std=Vpiv_std,
        Uptv_std=Uptv_std,
        Vptv_std=Vptv_std,
    )

    return


def generate_sst_pipeline_testing(root_folder):
    """
        Function to generate Tensorflow data pipeline for the testing dataset.

    :param root_folder:         String indicating the folder where the data is stored.
    :return dataset_test:       Tensorflow object containing the pipeline for the testing dataset.
    :return tot_samples_per_ds: Integer showing the total number of samples in the dataset
    """

    """
        Prepare files
    """

    # Define path where the tfrecord files are stored

    tfr_path = f"{root_folder}tfrecords/test/"

    # Select files in the tfrecord folder

    tfr_files = sorted([os.path.join(tfr_path,f) for f in os.listdir(tfr_path) if os.path.isfile(os.path.join(tfr_path,f))])

    # Keep only files terminated on .tfrecords

    regex = re.compile(f'.tfrecords')

    tfr_files = ([string for string in tfr_files if re.search(regex, string)])
   
    # Find number of samples for each tfrecord file
    
    n_samples_per_tfr = np.array([int(s.split('.')[-2][-2:].replace('_', '')) for s in tfr_files])
    n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]
    
    # Define total number of samples in the tfrecord files
    
    tot_samples_per_ds = sum(n_samples_per_tfr)

    # Initialize the testing dataset with the assigned tfrecord files. Internal shuffle is not applied

    tfr_files_test_ds = tf.data.Dataset.list_files(tfr_files, shuffle=False)

    # Parse the tfrecord files assigned to the testing dataset

    tfr_files_test_ds = tf.data.TFRecordDataset(tfr_files_test_ds)

    """
        Generated testing dataset pipeline
    """

    # Parse the information contained in the tfrecord files

    dataset_test = tfr_files_test_ds.map(lambda x: tf_parser_sst_testing(x, root_folder), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset_test, tot_samples_per_ds


def generate_sst_pipeline_training(root_folder, validation_split=0.2, shuffle_buffer=200, batch_size=8, n_prefetch=4):
    """
        Function to generate Tensorflow data pipelines for the training and validation datasets.

    :param root_folder:      String indicating the folder where the data is stored.
    :param validation_split: Float indicating the percentage of the training dataset to be used as validation dataset. Default is 0.2.
    :param shuffle_buffer:   Integer idicating the number of times to shuffle a file inside each tfrecord. Default is 200.
    :param batch_size:       Integer indicating the number of sample to be used during the stochastic gradient descent fitting of the neural network weights. Default is 8.
    :param n_prefecth:       Integer indicating the number of files preallocated during the training loop for each tfrecord. Default is 4.
    :return dataset_train:   Tensorflow object containing the pipeline for the training dataset.
    :return dataset_valid:   Tensorflow object containing the pipeline for the validation dataset.
    """

    """
        Prepare files
    """

    # Define path where the tfrecord files are stored

    tfr_path = f"{root_folder}tfrecords/train/"

    # Select files in the tfrecord folder

    tfr_files = sorted([os.path.join(tfr_path,f) for f in os.listdir(tfr_path) if os.path.isfile(os.path.join(tfr_path,f))])

    # Keep only files terminated on .tfrecords

    regex = re.compile(f'.tfrecords')

    tfr_files = ([string for string in tfr_files if re.search(regex, string)])

    """
        Separate files for training and validation
    """

    # Find number of samples for each tfrecord file
    
    n_samples_per_tfr = np.array([int(s.split('.')[-2][-2:]) for s in tfr_files])
    n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]

    # Define the cumulative number of samples in the tfrecord files

    cumulative_samples_per_tfr = np.cumsum(np.array(n_samples_per_tfr))

    # Define total number of samples in the tfrecord files

    tot_samples_per_ds = sum(n_samples_per_tfr)

    # Define number of tfrecord files available

    n_tfr_loaded_per_ds = int(tfr_files[0].split('_')[-3][-3:])

    # Ensure that the selected number of tfrecord files match the defined number of files available

    tfr_files = [string for string in tfr_files if int(string.split('_')[-3][:3]) <= n_tfr_loaded_per_ds]

    # Computer the number of files to be used in the training dataset

    n_samp_train = int(sum(n_samples_per_tfr) * (1 - validation_split))

    # Compute the number of file to be used in the validation dataset

    n_samp_valid = sum(n_samples_per_tfr) - n_samp_train

    # Define number of tfrecord files to be used for the training dataset and the  remaining number of files to be added from a shared tfrecord file with the validation dataset
    
    (n_files_train, samples_train_left) = np.divmod(n_samp_train, n_samples_per_tfr[0])

    # If there is shared files, add one to the number of training dataset files

    if samples_train_left > 0:

        n_files_train += 1

    # Select the tfrecord files for the training dataset

    tfr_files_train = [string for string in tfr_files if int(string.split('_')[-3][:3]) <= n_files_train]
    
    # Select number of tfrecord files that shared files between validation and training dataset (this is done because it is not mandatory that each tfrecord contains the same number of samples)

    n_tfr_left = np.sum(np.where(cumulative_samples_per_tfr < samples_train_left, 1, 0)) + 1
    
    # Define shared tfrecord files

    if sum([int(s.split('.')[-2][-2:]) for s in tfr_files_train]) != n_samp_train:

        # If the tfrecord files selected for the training dataset contains more samples than the number of training samples the last tfrecord file is shared with the validation dataset 
        shared_tfr = tfr_files_train[-1]
        tfr_files_valid = [shared_tfr]
    else:
        # If not, the validation dataset is initialized empty
        shared_tfr = ''
        tfr_files_valid = list()

    # The rest of tfrecord files not selected for the training dataset are added to the validation dataset

    tfr_files_valid.extend([string for string in tfr_files if string not in tfr_files_train])

    # Sort tfrecord files in the validation dataset
    tfr_files_valid = sorted(tfr_files_valid)

    # The shared tfrecord file is initialized as a TF constant

    shared_tfr_out = tf.constant(shared_tfr)

    # Initialize the number of tfrecord files as a TF constant

    n_tfr_per_ds = tf.constant(n_tfr_loaded_per_ds)

    # Initilize list containing the number of samples contained in each tfrecord file

    n_samples_loaded_per_tfr = list()

    # Assign files to the list

    if n_tfr_loaded_per_ds>1:

        # If there is more than one tfrecord file, the number of samples for each tfrecord file is added to the list except the last file

        n_samples_loaded_per_tfr.extend(n_samples_per_tfr[:n_tfr_loaded_per_ds-1])
        
        # The number of samples for the last tfrecord file is added as the difference between the toal number of samples and the cumulative number of samples until the penultimate file (REDUNDANT, it should be eliminated)

        n_samples_loaded_per_tfr.append(tot_samples_per_ds - cumulative_samples_per_tfr[n_tfr_loaded_per_ds-2])

    else:

        # If there is only one tfrecord file, add the total number of samples

        n_samples_loaded_per_tfr.append(tot_samples_per_ds)

    # Convert the list containing the number of samples per tfrecord file into a Numpy array

    n_samples_loaded_per_tfr = np.array(n_samples_loaded_per_tfr)

    # Initialize the training dataset with the assigned tfrexcord files. Seed is added to shuffle internally the tfrecord files
    
    tfr_files_train_ds = tf.data.Dataset.list_files(tfr_files_train, seed=666)

    # Initialize the validation dataset with the assigned tfrexcord files. Seed is added to shuffle internally the tfrecord files

    tfr_files_val_ds = tf.data.Dataset.list_files(tfr_files_valid, seed=686)

    # Check the number of shared tfrecord files

    if n_tfr_left>1:

        # If it is more than one, generate variable with the number of files in the shared tfrecord files for the training dataset

        samples_train_shared = samples_train_left - cumulative_samples_per_tfr[n_tfr_left-2]

        # Define the total number of samples in the shared tfrecord file

        n_samples_tfr_shared = n_samples_loaded_per_tfr[n_tfr_left-1]

    else:

        # If it is only one, rename the variable containing the files in the shared tfrecord file for the training dataset

        samples_train_shared = samples_train_left

        # Define the total number of samples in the shared tfrecord file

        n_samples_tfr_shared = n_samples_loaded_per_tfr[0]

    # Parse the tfrecord files assigned to the training dataset, excluding the samples belonging to the validation dataset

    # If the lambda file x, corresponding to the tfrecord file name, is equal to the shared tfrecord file name, take from that tfrecord file the first N number of samples, being N the number of samples in the shared tfrecord file assigned to the training dataset
    
    # If the lambda file x, corresponding to the tfrecord file name, contains samples belonging only to the training dataset, take the number of samples contained in that tfrecord file (check with the tfrecord file name that contains the information)

    tfr_files_train_ds = tfr_files_train_ds.interleave(
        lambda x : tf.data.TFRecordDataset(x).take(samples_train_shared) if tf.math.equal(x, shared_tfr_out) else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-3],sep='-')[0], tf.int32)-1)), 
        cycle_length=16, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Parse the tfrecord files assigned to the validation dataset, excluding the samples belonging to the training dataset

    # If the lambda file x, corresponding to the tfrecord file name, is equal to the shared tfrecord file name, take from that tfrecord file the last M number of samples, being M the difference between the total number of samples in the shared tfrecord file and the number of samples in the shared tfrecord file assigned to the training dataset

    # If the lambda file x, corresponding to the tfrecord file name, contains samples belonging only to the training dataset, take the number of samples contained in that tfrecord file (check with the tfrecord file name that contains the information)

    tfr_files_val_ds = tfr_files_val_ds.interleave(
        lambda x : tf.data.TFRecordDataset(x).skip(samples_train_shared).take(n_samples_tfr_shared - samples_train_shared) if tf.math.equal(x, shared_tfr_out) else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-3],sep='-')[0], tf.int32)-1)),
        cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    """
        Generated training dataset pipeline
    """

    # Parse the information contained in the tfrecord files

    dataset_train = tfr_files_train_ds.map(lambda x: tf_parser_sst(x, root_folder), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle the samples

    dataset_train = dataset_train.shuffle(shuffle_buffer)

    # Provide the pipeline with the batch size information

    dataset_train = dataset_train.batch(batch_size=batch_size)

    # Provide the pipeline with the number of prefetched files information

    dataset_train = dataset_train.prefetch(n_prefetch)

    """
        Generated validation dataset pipeline
    """

    # Parse the information contained in the tfrecord files

    dataset_valid = tfr_files_val_ds.map(lambda x: tf_parser_sst(x, root_folder), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle the samples

    dataset_valid = dataset_valid.shuffle(shuffle_buffer)

    # Provide the pipeline with the batch size information

    dataset_valid = dataset_valid.batch(batch_size=batch_size)

    # Provide the pipeline with the number of prefetched files information
    
    dataset_valid = dataset_valid.prefetch(n_prefetch)

    return dataset_train, dataset_valid


def generate_sst_scaling_data(root_folder, nx, ny, us, n_samples_train, n_samples_test):
    """
        Function to generate the scaling values from the training dataset.
    
    :param root_folder:        String indicating the folder where the data is stored.
    :param nx:                 Integer containing the grid points in the streamwise direction for the low-resolution data.
    :param ny:                 Integer containing the grid points in the wall-normal direction for the low-resolution data.
    :param us:                 Integer containing the upsampling ratio between the low- and high-resolution data.
    :param n_samples_train:    Integer indicating the number of samples in the training dataset.
    """


    Tpiv = np.zeros((n_samples_train, ny,    nx))
    Tptv = np.zeros((n_samples_train, ny*us, nx*us))
    Tdns = np.zeros((n_samples_train, ny*us, nx*us))
    Flag = np.zeros((n_samples_train, ny*us, nx*us))

    # Iterate over the number of samples

    for idx in tqdm(range(n_samples_train)):

        # Define path to Matlab file containing the actual sample

        try:

            piv_path = f"{root_folder}ss01/piv/SS1_{(idx+1):06d}.mat"
            ptv_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_{(idx+1):06d}.mat"
                
            # Load Matlab file

            piv = h5py.File(piv_path, 'r')
            ptv = h5py.File(ptv_path, 'r')

        except OSError:

            piv_path = f"{root_folder}ss01/piv/SS1_{(idx):06d}.mat"
            ptv_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_{(idx):06d}.mat"
                
            # Load Matlab file

            piv = h5py.File(piv_path, 'r')
            ptv = h5py.File(ptv_path, 'r')



        # Write each variable in their corresponding global matrices

        Tpiv[idx, :, :] = np.array(piv['T']).T
        Tptv[idx, :, :] = np.array(ptv['T']).T
        Flag[idx, :, :] = np.array(ptv['Flagptv']).T

    for idx in tqdm(range(n_samples_test)):

        # Define path to Matlab file containing the actual sample

        try:

            dns_path = f"{root_folder}ss{us:02d}/dns/DNS_SS{us}_{(12000+idx+1):06d}.mat"
                
            # Load Matlab file

            dns = h5py.File(dns_path, 'r')

        except OSError:

            dns_path = f"{root_folder}ss{us:02d}/dns/DNS_SS{us}_{(12000+idx):06d}.mat"
                
            # Load Matlab file

            dns = h5py.File(dns_path, 'r')



        # Write each variable in their corresponding global matrices

        Tpiv[idx, :, :] = np.array(piv['T']).T
        Tptv[idx, :, :] = np.array(ptv['T']).T
        Tdns[idx, :, :] = np.array(dns['PressDNS'])/np.array(dns['DensDNS']).T
        Flag[idx, :, :] = np.array(ptv['Flagptv']).T

    # Compute mean values

    Tpiv_mean = np.mean(Tpiv, axis=0)
    Tptv_mean = np.sum(Tptv, axis=0) / np.sum(Flag, axis=0)
    Tdns_mean = np.mean(Tdns, axis=0)

    
    Tpiv_std = np.std(Tpiv, axis=0)
    Tptv_std = np.sqrt(np.sum((Tptv - Tptv_mean*Flag)**2, axis=0) / np.sum(Flag, axis=0))
    Tdns_std = np.std(Tdns, axis=0)


    # Define path to stored the scaling values

    filename = f"{root_folder}ss{us:02}/tfrecords/scaling.npz"

    # Save scaling values

    np.savez(
        filename,
        Tpiv_mean=Tpiv_mean,
        Tptv_mean=Tptv_mean,
        Tdns_mean=Tdns_mean,
        Tpiv_std=Tpiv_std,
        Tptv_std=Tptv_std,
        Tdns_std=Tdns_std,
    )

    return


def generate_sst_tfrecords_testing(root_folder, us, n_samples_train, n_samples_test, max_samples_per_tf):
    """
        Function to generate tfrecords files for the training data.
    
    :param root_folder:        String indicating the folder where the data is stored.
    :param us:                 Integer containing the upsampling ratio between the low- and high-resolution data.
    :param n_samples_train:    Integer indicating the number of samples in the training dataset.
    :param max_samples_per_tf: Integer indicating the maximum number of samples that can be stored in a single tfrecord file.
    """
    
    bin_size = int(64/us)

    """
        Store tfrecord files
    """

    # Define path where the tfrecord files will be stored

    tfr_path = f"{root_folder}ss{us:02}/tfrecords/test/"

    # Check if the tfrecord folder exists

    if not os.path.exists(tfr_path):

        # If not, create it

        os.makedirs(tfr_path)

    # Define base for the tfrecords filename

    tfrecords_filename_base = f"{tfr_path}Nppp001_bin{bin_size}_ss{us}_PIVw64_PIVdx16_"

    # Calculate number of tfrecord files that will be generated

    n_sets = int(np.ceil(n_samples_test / max_samples_per_tf))

    # Define index indicating where the training sample should be acquired from

    idx = n_samples_train

    # Iterate over the calculated number of tfrecord files

    for n_set in tqdm(range(1, n_sets + 1)):

        # Check if this is the last tfrecord file

        if n_set * max_samples_per_tf < n_samples_test:

            # If not, store the maximum number of samples per tfrecord file

            num_smp = max_samples_per_tf

        else:

            # If this is the last tfrecord file store the remaining samples, always equal or smaller than the maximum number of samples per tfrecord file

            num_smp = n_samples_test - (n_set - 1) * max_samples_per_tf

        # Define filename for the actual tfrecord file

        tfrecords_filename = tfrecords_filename_base + f"file_{n_set:03d}-of-{n_sets:03d}_samples_{num_smp}.tfrecords"

        # Open Tensorflow writer object

        writer = tf.io.TFRecordWriter(tfrecords_filename)

        # Iterate over the number of samples to be stored in the actual tfrecord file

        for i in range(num_smp):

            # Check if this is the first sample
            
            piv_path = f"{root_folder}ss01/piv/SS1_{(idx+1):06d}.mat"
            ptv_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_{(idx+1):06d}.mat"
            dns_path = f"{root_folder}ss{us:02d}/dns/DNS_SS{us}_{(idx+1):06d}.mat"
            
            # Load Matlab file

            piv = h5py.File(piv_path, 'r')
        
            Tpiv = np.array(piv['T']).T
            
            ptv = h5py.File(ptv_path, 'r')

            # Generate matrices for each component
        
            Tptv = np.array(ptv['T']).T
            Flag = np.array(ptv['Flagptv']).T
            
            dns = h5py.File(dns_path, 'r')

            Tdns = np.array(dns['PressDNS']).T/np.array(dns['DensDNS']).T

            # import matplotlib.pyplot as plt

            # plt.subplot(121)
            # plt.imshow(Tpiv, vmin=Tpiv.min(), vmax=Tpiv.max())
            # plt.subplot(122)
            # plt.imshow(Tdns, vmin=Tpiv.min(), vmax=Tpiv.max())
            # plt.savefig('test.png')
            # jjj
            

            if idx == n_samples_train:

                # If so, allocate variable for the grid resolution in the high- and low-resolution data

                nx_piv = Tpiv.shape[1]
                ny_piv = Tpiv.shape[0]

                nx_ptv = Tptv.shape[1]
                ny_ptv = Tptv.shape[0]

                grid_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_grid.mat"

                grid = sio.loadmat(grid_path)

                xhr = grid['X']
                yhr = grid['Y']

                xlr = xhr[::us, ::us]
                ylr = yhr[::us, ::us]

            # Define sample to be stored

            example = tf.train.Example(
                features = tf.train.Features(
                    feature={
                        'i_sample': _int64_feature(idx),
                        'nx_piv': _int64_feature(int(nx_piv)),
                        'ny_piv': _int64_feature(int(ny_piv)),
                        'nx_ptv': _int64_feature(int(nx_ptv)),
                        'ny_ptv': _int64_feature(int(ny_ptv)),
                        'x_lr': _floatarray_feature(np.float32(xlr).flatten().tolist()),
                        'y_lr': _floatarray_feature(np.float32(ylr).flatten().tolist()),
                        'x_hr': _floatarray_feature(np.float32(xhr).flatten().tolist()),
                        'y_hr': _floatarray_feature(np.float32(yhr).flatten().tolist()),
                        'piv_raw1': _floatarray_feature(np.float32(Tpiv).flatten().tolist()),
                        # 'piv_raw2': _floatarray_feature(np.float32(Vpiv).flatten().tolist()),
                        'ptv_raw1': _floatarray_feature(np.float32(Tptv).flatten().tolist()),
                        # 'ptv_raw2': _floatarray_feature(np.float32(Vptv).flatten().tolist()),
                        'dns_raw1': _floatarray_feature(np.float32(Tdns).flatten().tolist()),
                        'ptv_flag': _floatarray_feature(np.float32(Flag).flatten().tolist()),
                    }
                )
            )  

            # Write sample in the tfrecord file

            writer.write(example.SerializeToString())

            # Advance the major loop one step

            idx += 1

        # Close Tensorflow writer object
        
        writer.close()

    return


def generate_sst_tfrecords_training(root_folder, us, n_samples_train, max_samples_per_tf):
    """
        Function to generate tfrecords files for the training data.
    
    :param root_folder:        String indicating the folder where the data is stored.
    :param us:                 Integer containing the upsampling ratio between the low- and high-resolution data.
    :param n_samples_train:    Integer indicating the number of samples in the training dataset.
    :param max_samples_per_tf: Integer indicating the maximum number of samples that can be stored in a single tfrecord file.
    """
    
    bin_size = int(64/us)

    """
        Store tfrecord files
    """

    # Define path where the tfrecord files will be stored

    tfr_path = f"{root_folder}ss{us:02}/tfrecords/train/"

    # Check if the tfrecord folder exists

    if not os.path.exists(tfr_path):

        # If not, create it

        os.makedirs(tfr_path)

    # Define base for the tfrecords filename

    tfrecords_filename_base = f"{tfr_path}Nppp001_bin{bin_size}_ss{us}_PIVw64_PIVdx16_"

    # Calculate number of tfrecord files that will be generated

    n_sets = int(np.ceil(n_samples_train / max_samples_per_tf))

    # Define index indicating where the training sample should be acquired from

    idx = 0

    # Iterate over the calculated number of tfrecord files

    for n_set in tqdm(range(1, n_sets + 1)):

        # Check if this is the last tfrecord file

        if n_set * max_samples_per_tf < n_samples_train:

            # If not, store the maximum number of samples per tfrecord file

            num_smp = max_samples_per_tf

        else:

            # If this is the last tfrecord file store the remaining samples, always equal or smaller than the maximum number of samples per tfrecord file

            num_smp = n_samples_train - (n_set - 1) * max_samples_per_tf

        # Define filename for the actual tfrecord file

        tfrecords_filename = tfrecords_filename_base + f"file_{n_set:03d}-of-{n_sets:03d}_samples_{num_smp}.tfrecords"

        # Open Tensorflow writer object

        writer = tf.io.TFRecordWriter(tfrecords_filename)

        # Iterate over the number of samples to be stored in the actual tfrecord file

        for i in range(num_smp):

            # Check if this is the first sample
            
            # piv_path = f"{root_folder}ss01/piv/PIV_{(idx+1):06d}.mat"
            piv_path = f"{root_folder}ss01/piv/SS1_{(idx+1):06d}.mat"
            ptv_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_{(idx+1):06d}.mat"
            
            # Load Matlab file
            try:
                piv = h5py.File(piv_path, 'r')
            
                Tpiv = np.array(piv['T']).T
                
                ptv = h5py.File(ptv_path, 'r')

                # Generate matrices for each component
            
                Tptv = np.array(ptv['T']).T
                Flag = np.array(ptv['Flagptv']).T

            except OSError:

                piv_path = f"{root_folder}ss01/piv/SS1_{(idx):06d}.mat"
                ptv_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_{(idx):06d}.mat"
                piv = h5py.File(piv_path, 'r')
            
                Tpiv = np.array(piv['T']).T
                
                ptv = h5py.File(ptv_path, 'r')

                # Generate matrices for each component
            
                Tptv = np.array(ptv['T']).T
                Flag = np.array(ptv['Flagptv']).T

                print('OSError')


 
            if idx == 0:

                # If so, allocate variable for the grid resolution in the high- and low-resolution data

                nx_piv = Tpiv.shape[1]
                ny_piv = Tpiv.shape[0]

                nx_ptv = Tptv.shape[1]
                ny_ptv = Tptv.shape[0]

                # grid_path = f"{root_folder}ss01/piv/SS1_grid.mat"
                # grid = h5py.File(grid_path, 'r')
                # print(grid.keys())
                # kkk

                # xlr = np.array(piv['XPIV']).T
                # ylr = np.array(piv['YPIV']).T

                # xhr = xlr * us
                # yhr = ylr * us

            # Define sample to be stored

            example = tf.train.Example(
                features = tf.train.Features(
                    feature={
                        'i_sample': _int64_feature(idx),
                        'nx_piv': _int64_feature(int(nx_piv)),
                        'ny_piv': _int64_feature(int(ny_piv)),
                        'nx_ptv': _int64_feature(int(nx_ptv)),
                        'ny_ptv': _int64_feature(int(ny_ptv)),
                        # 'x_lr': _floatarray_feature(np.float32(xlr).flatten().tolist()),
                        # 'y_lr': _floatarray_feature(np.float32(ylr).flatten().tolist()),
                        # 'x_hr': _floatarray_feature(np.float32(xhr).flatten().tolist()),
                        # 'y_hr': _floatarray_feature(np.float32(yhr).flatten().tolist()),
                        'piv_raw1': _floatarray_feature(np.float32(Tpiv).flatten().tolist()),
                        'ptv_raw1': _floatarray_feature(np.float32(Tptv).flatten().tolist()),
                        'ptv_flag': _floatarray_feature(np.float32(Flag).flatten().tolist()),
                    }
                )
            )  

            # Write sample in the tfrecord file

            writer.write(example.SerializeToString())

            # Advance the major loop one step

            idx += 1

        # Close Tensorflow writer object
        
        writer.close()

    return


def generate_tfrecords_training_exptbl(root_folder, us, n_samples_train, max_samples_per_tf):
    """
        Function to generate tfrecords files for the training data.
    
    :param root_folder:        String indicating the folder where the data is stored.
    :param us:                 Integer containing the upsampling ratio between the low- and high-resolution data.
    :param n_samples_train:    Integer indicating the number of samples in the training dataset.
    :param max_samples_per_tf: Integer indicating the maximum number of samples that can be stored in a single tfrecord file.
    """
    
    bin_size = int(64/us)

    """
        Store tfrecord files
    """

    # Define path where the tfrecord files will be stored

    tfr_path = f"{root_folder}ss{us:02}/tfrecords/train/"

    # Check if the tfrecord folder exists

    if not os.path.exists(tfr_path):

        # If not, create it

        os.makedirs(tfr_path)

    # Define base for the tfrecords filename

    tfrecords_filename_base = f"{tfr_path}Nppp001_bin{bin_size}_ss{us}_PIVw64_PIVdx16_"

    # Calculate number of tfrecord files that will be generated

    n_sets = int(np.ceil(n_samples_train / max_samples_per_tf))

    # Define index indicating where the training sample should be acquired from

    idx = 0

    # Iterate over the calculated number of tfrecord files

    for n_set in tqdm(range(1, n_sets + 1)):

        # Check if this is the last tfrecord file

        if n_set * max_samples_per_tf < n_samples_train:

            # If not, store the maximum number of samples per tfrecord file

            num_smp = max_samples_per_tf

        else:

            # If this is the last tfrecord file store the remaining samples, always equal or smaller than the maximum number of samples per tfrecord file

            num_smp = n_samples_train - (n_set - 1) * max_samples_per_tf

        # Define filename for the actual tfrecord file

        tfrecords_filename = tfrecords_filename_base + f"file_{n_set:03d}-of-{n_sets:03d}_samples_{num_smp}.tfrecords"

        # Open Tensorflow writer object

        writer = tf.io.TFRecordWriter(tfrecords_filename)

        # Iterate over the number of samples to be stored in the actual tfrecord file

        for i in range(num_smp):

            # Check if this is the first sample
            
            piv_path = f"{root_folder}ss01/piv/PIV_{(idx+1):06d}.mat"
            ptv_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_{(idx+1):06d}.mat"
            
            # Load Matlab file

            piv = h5py.File(piv_path, 'r')
        
            Upiv = np.array(piv['U']).T[:, 1:]
            Vpiv = np.array(piv['V']).T[:, 1:]
            
            ptv = h5py.File(ptv_path, 'r')

            # Generate matrices for each component
        
            Uptv = np.array(ptv['Uptv']).T[:, us:]
            Vptv = np.array(ptv['Vptv']).T[:, us:]
            Flag = np.array(ptv['Flagptv']).T[:, us:]

            if idx == 0:

                # If so, allocate variable for the grid resolution in the high- and low-resolution data

                nx_piv = Upiv.shape[1]
                ny_piv = Upiv.shape[0]

                nx_ptv = Uptv.shape[1]
                ny_ptv = Uptv.shape[0]

                xlr = np.array(piv['XPIV']).T[:,1:]
                ylr = np.array(piv['YPIV']).T[:,1:]

                grid_path = f"{root_folder}ss{us:02d}/ptv/SS{us}_grid.mat"

                grid = sio.loadmat(grid_path)

                xhr = grid['X'][:, us:]
                yhr = grid['Y'][:, us:]

            # Define sample to be stored

            example = tf.train.Example(
                features = tf.train.Features(
                    feature={
                        'i_sample': _int64_feature(idx),
                        'nx_piv': _int64_feature(int(nx_piv)),
                        'ny_piv': _int64_feature(int(ny_piv)),
                        'nx_ptv': _int64_feature(int(nx_ptv)),
                        'ny_ptv': _int64_feature(int(ny_ptv)),
                        'x_lr': _floatarray_feature(np.float32(xlr).flatten().tolist()),
                        'y_lr': _floatarray_feature(np.float32(ylr).flatten().tolist()),
                        'x_hr': _floatarray_feature(np.float32(xhr).flatten().tolist()),
                        'y_hr': _floatarray_feature(np.float32(yhr).flatten().tolist()),
                        'piv_raw1': _floatarray_feature(np.float32(Upiv).flatten().tolist()),
                        'piv_raw2': _floatarray_feature(np.float32(Vpiv).flatten().tolist()),
                        'ptv_raw1': _floatarray_feature(np.float32(Uptv).flatten().tolist()),
                        'ptv_raw2': _floatarray_feature(np.float32(Vptv).flatten().tolist()),
                        'ptv_flag': _floatarray_feature(np.float32(Flag).flatten().tolist()),
                    }
                )
            )  

            # Write sample in the tfrecord file

            writer.write(example.SerializeToString())

            # Advance the major loop one step

            idx += 1

        # Close Tensorflow writer object
        
        writer.close()

    return


@tf.function
def tf_parser_old(rec, root_folder):
    """
        Function to parse the information contained in the tfrecord files.

    :param rec:         Binary file containing the information for one sample.
    :param root_folder: String containg the folder where data is stored
    :return piv:        Tensorflow array with the Particle Image Velocimetry information (wall-normal x streamwise x channels)
    :return ptv:        Tensorflow array with the Particle Tracking Velocimetry information (wall-normal x streamwise x channels)
    :return dns:        Tensorflow array with the Direct Numerical Simulation information (wall-normal x streamwise x channels)
    :return flag:       Tensorflow array with the number of PTV grid point containing information (wall-normal x streamwise x channels)
    """

    """
        Read data
    """
    
    # Define dictionary for the variables' name and type contained in the binary files 

    features = {
        'i_sample': tf.io.FixedLenFeature([], tf.int64),                               # Sample number in the DNS extraction procedure
        'nx_piv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the streamwise direction for the low-resolution data 
        'ny_piv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the wall-normal direction for the low-resolution data 
        'nx_ptv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the streamwise direction for the high-resolution data 
        'ny_ptv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the wall-normal direction for the high-resolution data 
        'x_lr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # Low-resolution domain length in the streamwise direction
        'y_lr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # Low-resolution domain length in the wall-normal direction
        'x_hr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # High-resolution domain length in the streamwise direction
        'y_hr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # High-resolution domain length in the wall-normal direction
        'piv_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the low-resolution data
        'piv_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Wall-normal velocity in the low-resolution data
        'ptv_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the high-resolution data
        'ptv_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Wall-normal velocity in the high-resolution data
        'dns_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the original data
        'dns_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Wall-normal velocity in the original data
        'ptv_flag': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Bins in the high-resolution data containing information
    }

    # Parse binary file

    parsed_rec = tf.io.parse_single_example(rec, features)

    # Read sample number in the DNS extraction procedure

    i_smp = tf.cast(parsed_rec['i_sample'], tf.int32)

    # Read streamwise and wall-normal grid points inthe low-resolution data

    nx_piv = tf.cast(parsed_rec['nx_piv'], tf.int32)
    ny_piv = tf.cast(parsed_rec['ny_piv'], tf.int32)

    # Read streamwise and wall-normal grid points inthe high-resolution data

    nx_ptv = tf.cast(parsed_rec['nx_ptv'], tf.int32)
    ny_ptv = tf.cast(parsed_rec['ny_ptv'], tf.int32)

    """
        Scale data
    """

    # Define path to file containing scaling value

    filename = f"{root_folder}tfrecords/scaling.npz"

    # Load mean velocity values in the streamwise and wall-normal directions for low- and high-resolution data

    Upiv_mean = np.expand_dims(np.load(filename)['Upiv_mean'], axis=2)
    Vpiv_mean = np.expand_dims(np.load(filename)['Vpiv_mean'], axis=2)
    Uptv_mean = np.expand_dims(np.load(filename)['Uptv_mean'], axis=2)
    Vptv_mean = np.expand_dims(np.load(filename)['Vptv_mean'], axis=2)

    # Load standard deviation velocity values in the streamwise and wall-normal directions for low- and high-resolution data

    Upiv_std = np.expand_dims(np.load(filename)['Upiv_std'], axis=2)
    Vpiv_std = np.expand_dims(np.load(filename)['Vpiv_std'], axis=2)
    Uptv_std = np.expand_dims(np.load(filename)['Uptv_std'], axis=2)
    Vptv_std = np.expand_dims(np.load(filename)['Vptv_std'], axis=2)

    # Reshape flag data into 2-dimensional matrix 

    flag = tf.reshape(parsed_rec['ptv_flag'], (ny_ptv, nx_ptv, 1))

    # Reshape data into 2-dimensional matrix, substract mean value and divide by the standard deviation. Concatenate the streamwise and wall-normal velocities along the third dimension

    piv = (tf.reshape(parsed_rec['piv_raw1'], (ny_piv, nx_piv, 1)) - Upiv_mean) / Upiv_std
    piv = tf.concat((piv, (tf.reshape(parsed_rec['piv_raw2'], (ny_piv, nx_piv, 1)) - Vpiv_mean) / Vpiv_std), -1)

    ptv = (tf.reshape(parsed_rec['ptv_raw1'], (ny_ptv, nx_ptv, 1)) - Uptv_mean * flag) / Uptv_std * flag
    ptv = tf.concat((ptv, (tf.reshape(parsed_rec['ptv_raw2'], (ny_ptv, nx_ptv, 1)) - Vptv_mean * flag) / Vptv_std * flag), -1)

    dns = tf.reshape(parsed_rec['dns_raw1'], (ny_ptv, nx_ptv, 1))
    dns = tf.concat((dns, tf.reshape(parsed_rec['dns_raw2'], (ny_ptv, nx_ptv, 1))), -1)

    return piv, ptv, dns, flag




@tf.function
def tf_parser_sst(rec, root_folder):
    """
        Function to parse the information contained in the tfrecord files.

    :param rec:         Binary file containing the information for one sample.
    :param root_folder: String containg the folder where data is stored
    :return piv:        Tensorflow array with the Particle Image Velocimetry information (wall-normal x streamwise x channels)
    :return ptv:        Tensorflow array with the Particle Tracking Velocimetry information (wall-normal x streamwise x channels)
    :return dns:        Tensorflow array with the Direct Numerical Simulation information (wall-normal x streamwise x channels)
    :return flag:       Tensorflow array with the number of PTV grid point containing information (wall-normal x streamwise x channels)
    """

    """
        Read data
    """
    
    # Define dictionary for the variables' name and type contained in the binary files 

    features = {
        'i_sample': tf.io.FixedLenFeature([], tf.int64),                               # Sample number in the DNS extraction procedure
        'nx_piv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the streamwise direction for the low-resolution data 
        'ny_piv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the wall-normal direction for the low-resolution data 
        'nx_ptv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the streamwise direction for the high-resolution data 
        'ny_ptv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the wall-normal direction for the high-resolution data 
        'x_lr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # Low-resolution domain length in the streamwise direction
        'y_lr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # Low-resolution domain length in the wall-normal direction
        'x_hr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # High-resolution domain length in the streamwise direction
        'y_hr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # High-resolution domain length in the wall-normal direction
        'piv_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the low-resolution data
        'ptv_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the high-resolution data
        'ptv_flag': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Bins in the high-resolution data containing information
    }

    # Parse binary file

    parsed_rec = tf.io.parse_single_example(rec, features)

    # Read sample number in the DNS extraction procedure

    i_smp = tf.cast(parsed_rec['i_sample'], tf.int32)

    # Read streamwise and wall-normal grid points inthe low-resolution data

    nx_piv = tf.cast(parsed_rec['nx_piv'], tf.int32)
    ny_piv = tf.cast(parsed_rec['ny_piv'], tf.int32)

    # Read streamwise and wall-normal grid points inthe high-resolution data

    nx_ptv = tf.cast(parsed_rec['nx_ptv'], tf.int32)
    ny_ptv = tf.cast(parsed_rec['ny_ptv'], tf.int32)

    """
        Scale data
    """

    # Define path to file containing scaling value

    filename = f"{root_folder}tfrecords/scaling.npz"

    # Load mean velocity values in the streamwise and wall-normal directions for low- and high-resolution data

    Tpiv_mean = np.expand_dims(np.load(filename)['Tpiv_mean'], axis=2)
    Tptv_mean = np.expand_dims(np.load(filename)['Tptv_mean'], axis=2)

    # Load standard deviation velocity values in the streamwise and wall-normal directions for low- and high-resolution data

    Tpiv_std = np.expand_dims(np.load(filename)['Tpiv_std'], axis=2)
    Tptv_std = np.expand_dims(np.load(filename)['Tptv_std'], axis=2)

    # Reshape flag data into 2-dimensional matrix 

    flag = tf.reshape(parsed_rec['ptv_flag'], (ny_ptv, nx_ptv, 1))

    # Reshape data into 2-dimensional matrix, substract mean value and divide by the standard deviation. Concatenate the streamwise and wall-normal velocities along the third dimension

    piv = (tf.reshape(parsed_rec['piv_raw1'], (ny_piv, nx_piv, 1)) - Tpiv_mean) / Tpiv_std
    piv = tf.where(tf.math.is_nan(piv), tf.zeros_like(piv), piv)
    ptv = (tf.reshape(parsed_rec['ptv_raw1'], (ny_ptv, nx_ptv, 1)) - Tptv_mean * flag) / Tptv_std
    ptv = tf.where(tf.math.is_nan(ptv), tf.zeros_like(ptv), ptv)

    return piv, ptv, flag



@tf.function
def tf_parser_sst_testing(rec, root_folder):
    """
        Function to parse the information contained in the tfrecord files.

    :param rec:         Binary file containing the information for one sample.
    :param root_folder: String containg the folder where data is stored
    :return piv:        Tensorflow array with the Particle Image Velocimetry information (wall-normal x streamwise x channels)
    :return ptv:        Tensorflow array with the Particle Tracking Velocimetry information (wall-normal x streamwise x channels)
    :return dns:        Tensorflow array with the Direct Numerical Simulation information (wall-normal x streamwise x channels)
    :return flag:       Tensorflow array with the number of PTV grid point containing information (wall-normal x streamwise x channels)
    """

    """
        Read data
    """
    
    # Define dictionary for the variables' name and type contained in the binary files 

    features = {
        'i_sample': tf.io.FixedLenFeature([], tf.int64),                               # Sample number in the DNS extraction procedure
        'nx_piv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the streamwise direction for the low-resolution data 
        'ny_piv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the wall-normal direction for the low-resolution data 
        'nx_ptv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the streamwise direction for the high-resolution data 
        'ny_ptv': tf.io.FixedLenFeature([], tf.int64),                                 # Number of grid points in the wall-normal direction for the high-resolution data 
        'x_lr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # Low-resolution domain length in the streamwise direction
        'y_lr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # Low-resolution domain length in the wall-normal direction
        'x_hr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # High-resolution domain length in the streamwise direction
        'y_hr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     # High-resolution domain length in the wall-normal direction
        'piv_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the low-resolution data
        'ptv_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the high-resolution data
        'dns_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Streamwise velocity in the high-resolution data
        'ptv_flag': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), # Bins in the high-resolution data containing information
    }

    # Parse binary file

    parsed_rec = tf.io.parse_single_example(rec, features)

    # Read sample number in the DNS extraction procedure

    i_smp = tf.cast(parsed_rec['i_sample'], tf.int32)

    # Read streamwise and wall-normal grid points inthe low-resolution data

    nx_piv = tf.cast(parsed_rec['nx_piv'], tf.int32)
    ny_piv = tf.cast(parsed_rec['ny_piv'], tf.int32)

    # Read streamwise and wall-normal grid points inthe high-resolution data

    nx_ptv = tf.cast(parsed_rec['nx_ptv'], tf.int32)
    ny_ptv = tf.cast(parsed_rec['ny_ptv'], tf.int32)

    """
        Scale data
    """

    # Define path to file containing scaling value

    filename = f"{root_folder}tfrecords/scaling.npz"

    # Load mean velocity values in the streamwise and wall-normal directions for low- and high-resolution data

    Tpiv_mean = np.expand_dims(np.load(filename)['Tpiv_mean'], axis=2)
    Tptv_mean = np.expand_dims(np.load(filename)['Tptv_mean'], axis=2)

    # Load standard deviation velocity values in the streamwise and wall-normal directions for low- and high-resolution data

    Tpiv_std = np.expand_dims(np.load(filename)['Tpiv_std'], axis=2)
    Tptv_std = np.expand_dims(np.load(filename)['Tptv_std'], axis=2)

    # Reshape flag data into 2-dimensional matrix 

    flag = tf.reshape(parsed_rec['ptv_flag'], (ny_ptv, nx_ptv, 1))

    # Reshape data into 2-dimensional matrix, substract mean value and divide by the standard deviation. Concatenate the streamwise and wall-normal velocities along the third dimension

    piv = (tf.reshape(parsed_rec['piv_raw1'], (ny_piv, nx_piv, 1)) - Tpiv_mean) / Tpiv_std
    
    ptv = (tf.reshape(parsed_rec['ptv_raw1'], (ny_ptv, nx_ptv, 1)) - Tptv_mean * flag) / Tptv_std

    dns = tf.reshape(parsed_rec['dns_raw1'], (ny_ptv, nx_ptv, 1))

    xlr = tf.reshape(parsed_rec['x_lr'], (ny_piv, nx_piv))
    ylr = tf.reshape(parsed_rec['y_lr'], (ny_piv, nx_piv))
    xhr = tf.reshape(parsed_rec['x_hr'], (ny_ptv, nx_ptv))
    yhr = tf.reshape(parsed_rec['y_hr'], (ny_ptv, nx_ptv))

    return piv, ptv, dns, flag, xlr, ylr, xhr, yhr

>>>>>>> d697111dba0b1b8df36000c0c120a92bf7e61e0e
