# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 13:26:04 2021
@author: guemesturb
"""


import os
import time
import tensorflow as tf


def training_loop(root_folder, model_name, dataset_train, dataset_valid, generator, discriminator, generator_loss, discriminator_loss, generator_optimizer, discriminator_optimizer, pretrained=False, epochs=100, saving_freq=5):
    """
        Training logic for GAN architectures.

    :param root_folder:             String containg the folder where data is stored
    :param model_name:              String containing the assigned name to the model, for storage purposes.
    :param dataset_train:           Tensorflow pipeline for the training dataset.
    :param dataset_valid:           Tensorflow pipeline for the validation dataset.
    :param generator:               Tensorflow object containing the genertaor architecture.
    :param discriminator:           Tensorflow object containing the discriminator architecture.
    :param generator loss:          Tensorflow object containing the loss function for the generator fitting.
    :param discriminator loss:      Tensorflow object containing the loss function for the discriminator fitting.
    :param generator_optimizer:     Tensorflow object containing the optimizer for the generator architecture.
    :param discriminator_optimizer: Tensorflow object containing the optimizer for the discriminator arquitecture.
    :param pretrained:              Boolean flag indicating whether there exists a trained mordel from which restart the training or not. Default is False.
    :param epochs:                  Integer the number of epochs to train the models. Default is 100.
    :param saving_freq:             Integer containing the saving frequency during the training loop. Default is every 5 epochs.
    :return:
    """

    """
        Define folder for logging
    """

    # Define log path

    log_folder = f"{root_folder}logs/"

    # Check if exists the log folder

    if not os.path.exists(log_folder):

        # If it does not exist, create it

        os.mkdir(log_folder)

    """
        Define checkpoint object
    """

    # Define path to checkpoint directory

    checkpoint_dir = f"{root_folder}models/checkpoints_{model_name}"

    # Check if exists the checpoint folder

    if not os.path.exists(checkpoint_dir):

        # If it does not exist, create it

        os.makedirs(checkpoint_dir)

    # Define checkpoint prefix

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    # Generate checkpoint object to track the generator and discriminator architectures and optimizers

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )

    """
        Prepare training
    """

    # Check if there exists an already trained model

    if pretrained:

        # If it exists, restore the checkpoint with the last stored state

        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        # Open log file of the pretrained model

        with open(f"{log_folder}/log_{model_name}.log",'r') as fd:

            # Read the number of the last epoch trainined

            epoch_bias = int(fd.read().splitlines()[-1].split(',')[0])

        # Compute the last epoch number for which the models were stored, also known as epoch bias

        epoch_bias = epoch_bias // saving_freq * saving_freq

        # Compute the last epochs that were not stored due to the saving frequency

        drop = epoch_bias % saving_freq

        # Check whether there were epochs not stored or not

        if drop != 0:

            # From the pretrained log file, keep only the epochs that were stored

            with open(f"{log_folder}/log_{model_name}.log", 'r') as fd:
            
                lines = fd.readlines()[:-drop]

            # Regenerate the log file to delete the non-stored epochs

            with open(f"{log_folder}/log_{model_name}.log", 'w') as fd:

                fd.writelines(lines)

        print(f"Training reinitialized after {epoch_bias} epochs")

    else:

        # If not, generate a new log file

        with open(f"{log_folder}/log_{model_name}.log",'w') as fd:

            fd.write(f"epoch,gen_loss,disc_loss,val_gen_loss,val_disc_loss,time\n")

        # Define the epoch bias as zero

        epoch_bias = 0

    # Define object to track the training and validation losses during the training loop

    train_gen_loss = tf.metrics.Mean()
    train_disc_loss = tf.metrics.Mean()
    valid_gen_loss = tf.metrics.Mean()
    valid_disc_loss = tf.metrics.Mean()

    """
        Training loop
    """

    # Define start time

    start_time = time.time()

    # Enter in loop

    for epoch in range(epoch_bias + 1, epoch_bias + epochs + 1):

        # Reset to zero losses values in every epoch

        train_gen_loss.reset_states()
        train_disc_loss.reset_states()
        valid_gen_loss.reset_states()
        valid_disc_loss.reset_states()

        # Iterate over the training dataset 
        
        for (lr_target, hr_target, fl_target) in dataset_train:

            # Update the models' weights and compute training losses

            gen_loss, disc_loss = train_step(lr_target, hr_target, fl_target, generator, discriminator, generator_loss, discriminator_loss, generator_optimizer, discriminator_optimizer)

            # Update training losses object
                    
            train_gen_loss.update_state(gen_loss)
            train_disc_loss.update_state(disc_loss)

        # Iterate over the validation dataset

        for (lr_target, hr_target, fl_target) in dataset_valid:

            # Compute validation losses

            gen_loss, disc_loss = valid_step(lr_target, hr_target, fl_target, generator, discriminator, generator_loss, discriminator_loss)

            # Update validation losses object
                    
            valid_gen_loss.update_state(gen_loss)
            valid_disc_loss.update_state(disc_loss)

        # Define end time for iteration
                
        end_time = time.time()

        # Check if the model needs to be saved in this epoch

        if epoch % saving_freq == 0:

            # Save checkpoint status

            checkpoint.save(file_prefix = checkpoint_prefix)

        # Open log file

        with open(f'{log_folder}/log_{model_name}.log','a') as fd:

            # Write epoch information in log file

            fd.write(f"{epoch},{train_gen_loss.result().numpy():0.5f},{train_disc_loss.result().numpy():0.5f},{valid_gen_loss.result().numpy():0.5f},{valid_disc_loss.result().numpy():0.5f},{(end_time - start_time):0.5f}\n")

        # Print epoch information in terminal

        print(f'Epoch {epoch:04d}/{epochs:04d}, gen_loss: {train_gen_loss.result().numpy():0.5f}, disc_loss: {train_disc_loss.result().numpy():0.5f}, val_gen_loss: {valid_gen_loss.result().numpy():0.5f}, val_disc_loss: {valid_disc_loss.result().numpy():0.5f}, elapsed time from start: {(end_time - start_time):0.5f}')

    return


@tf.function
def train_step(lr_target, hr_target, fl_target, generator, discriminator, generator_loss, discriminator_loss, generator_optimizer, discriminator_optimizer):
    """
        Function to compute the train step, updating the models' weights.
              
    :param lr_target:               Tensorflow array with the target low-resolution data.
    :param hr_target:               Tensorflow array with the target high-resolution data.
    :param fl_target:               Tensorflow array with the target bin-availabity information for high-resolution data
    :param generator:               Tensorflow object containing the genertaor architecture.
    :param discriminator:           Tensorflow object containing the discriminator architecture.
    :param generator loss:          Tensorflow object containing the loss function for the generator fitting.
    :param discriminator loss:      Tensorflow object containing the loss function for the discriminator fitting.
    :param generator_optimizer:     Tensorflow object containing the optimizer for the generator architecture.
    :param discriminator_optimizer: Tensorflow object containing the optimizer for the discriminator arquitecture.
    :return gen_los:                Float containing the generator loss.
    :return disc_los:               Float containing the discriminator loss.
    """

    # Define tape for automatic differentiation
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Compute generator's prediction

        hr_predic = generator(lr_target, training=True)

        # Compute discriminator's prediction for target and predicted high-resolution data
        
        real_ptv = discriminator(hr_target, training=True)
        fake_ptv = discriminator(tf.math.multiply(hr_predic, fl_target), training=True)

        # Compute generator and discriminator losses

        gen_loss = generator_loss(fake_ptv, hr_predic, hr_target, fl_target)
        disc_loss = discriminator_loss(real_ptv, fake_ptv)

    # Compute gradients for the generator and discriminator
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Update the generator and discriminator weights
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss


@tf.function
def valid_step(lr_target, hr_target, fl_target, generator, discriminator, generator_loss, discriminator_loss):
    """
        Function to compute the train step, updating the models' weights.
              
    :param lr_target:               Tensorflow array with the target low-resolution data.
    :param hr_target:               Tensorflow array with the target high-resolution data.
    :param fl_target:               Tensorflow array with the target bin-availabity information for high-resolution data
    :param generator:               Tensorflow object containing the genertaor architecture.
    :param discriminator:           Tensorflow object containing the discriminator architecture.
    :param generator loss:          Tensorflow object containing the loss function for the generator fitting.
    :param discriminator loss:      Tensorflow object containing the loss function for the discriminator fitting.
    :return gen_los:                Float containing the generator loss.
    :return disc_los:               Float containing the discriminator loss.
    """

    # Define tape for automatic differentiation
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Compute generator's prediction

        hr_predic = generator(lr_target, training=False)

        # Compute discriminator's prediction for target and predicted high-resolution data
        
        real_ptv = discriminator(hr_target, training=False)
        fake_ptv = discriminator(tf.math.multiply(hr_predic, fl_target), training=False)

        # Compute generator and discriminator losses

        gen_loss = generator_loss(fake_ptv, hr_predic, hr_target, fl_target)
        disc_loss = discriminator_loss(real_ptv, fake_ptv)
    
    return gen_loss, disc_loss

