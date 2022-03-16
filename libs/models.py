# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 11:38:43 2021
@author: guemesturb
"""


import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def SubpixelConv2D(input_shape, scale=4):
    """
        Custom layer to shuffle abd reduce tensor filters (last dimension) to upsample their height and width.

    :param input_shape: Tensorflow object containing the tensor size (batch x height x width x filters).
    :param scale: Integer containing the ratio to increase/decrease the heigth and width/number of filters.
    :return: Tensorflow tensor with rescaled dimensions
    """

    def subpixel_shape(input_shape):
        """
            Function to compute the new tensor size after pixel shuffling.

        :param input_shape: Tensorflow object containing the tensor size (batch x height x width x filters).
        :return: Tuple containing the rescaled tensor size (batch x height x width x filters).
        """
        
        # Compute new dimensions

        dims = [input_shape[0], input_shape[1] * scale, input_shape[2] * scale, int(input_shape[3] / (scale ** 2))]

        # Transform list into tuple

        output_shape = tuple(dims)
        
        return output_shape

    def subpixel(x):
        """
            Function to change tensor size.

        :return: Tensorflow tensor with rescaled dimensions
        """

        return tf.nn.depth_to_space(x, scale, data_format='NHWC')

    return layers.Lambda(subpixel, output_shape=subpixel_shape)


class GANPIV(object):
  
    def __init__(self, model_name, us, nx, ny, channels=2, n_residual_blocks=16):
        """
            Python class to generate the Tensorflow models for resolution enhancement of PIV images through GANs.

        :param model_name:         String containing the assigned name to the model, for storage purposes.
        :param us:                 Integer containing the upsampling ratio between the low- and high-resolution data.
        :param nx:                 Integer containing the grid points in the streamwise direction for the low-resolution data.
        :param ny:                 Integer containing the grid points in the wall-normal direction for the low-resolution data.
        :param channels:           Integer containing the number of velocity components present in the data. Default is 2.
        :param n_residual_blocks : Integer containing the number residual blocks to be applied in the GAN generator. Default is 16.
        :return:
        """

        # Declare variable inside the class
        
        self.us = us
        self.nx = nx
        self.ny = ny
        self.channels = channels
        self.model_name = model_name
        self.n_residual_blocks = n_residual_blocks

        return


    def build_architecture(self):

        if (self.model_name == 'architecture-01') or (self.model_name == 'architecture01'):

            return self.architecture01()

        else:

            return self.architecture01()


    def architecture01(self):
        """
            Function to generate the SRGAN architecture as Ledig et al. (2017). This version is modified with respect the original one by removing the batch normalization layers.

        :return generator:          Tensorflow object containing the generator model.
        :return discriminator:      Tensorflow object containing the discriminator model.
        :return generator_loss:     Self-defined Python function containing the generator loss
        :return discriminator_loss: Self-defined Python function containing the discriminator loss.
        """

        """
            Generator model
        """

        def res_block_gen(model, kernal_size, filters, strides):
            """
                Function to generate a residual block

            :param model:       Tensorflow tensor containing the internal model state.
            :param kernel_size: Integer containing the kernel (or filter) size for the convolutional operations.
            :param filters:     Integer containing the number of filters to apply in the convolution operation.
            :param strides:     Integer containing the stride value to ba applied in the convolutional operations.
            :return model:      Tensorflow tensor
            """

            # Copy model for skip-connection purposes

            gen = model

            # Apply convolutional operation

            model = layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_last')(model)

            # Apply Parametric ReLU activation function

            model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)

            # Apply convolutional operation

            model = layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_last')(model) 

            # Add model with input model (skip connection operation)

            model = layers.Add()([gen, model])
            
            return model

        def up_sampling_block(model, kernel_size, filters, strides):
            """
                Function to upsample the wight and height dimensions of the data.

            :param model:       Tensorflow tensor containing the internal model state.
            :param kernel_size: Integer containing the kernel (or filter) size for the convolutional operations.
            :param filters:     Integer containing the number of filters to apply in the convolution operation.
            :param strides:     Integer containing the stride value to ba applied in the convolutional operations.
            :return model:      Tensorflow tensor
            """

            # Apply convolutional operation

            model = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_last')(model)

            # Apply Pixxle Shuffle layer

            model = SubpixelConv2D(model.shape, scale=2)(model)

            # Apply Parametric ReLU activation function

            model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
            
            return model

        # Define input layer

        inputs = keras.Input(shape=(self.ny, self.nx, self.channels), name='low-res-input')

        # Apply a convolutional layer

        conv_1 = layers.Conv2D(filters=64, kernel_size=9, strides=1, activation='linear', data_format='channels_last', padding='same')(inputs)
        # conv_1 = layers.Conv2D(filters=128, kernel_size=9, strides=1, activation='linear', data_format='channels_last', padding='same')(inputs)

        # Apply a Parametric ReLU activation function

        prelu_1 = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(conv_1)

        # Copy model for its use in a residual-block loop

        res_block = prelu_1

        # Apply N residual blocks

        for index in range(self.n_residual_blocks):

            res_block = res_block_gen(res_block, 3, 64, 1)
            # res_block = res_block_gen(res_block, 3, 128, 1)
        
        # Apply a convolutional layer

        conv_2 = layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same", data_format='channels_last')(res_block)
        # conv_2 = layers.Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same", data_format='channels_last')(res_block)

        # Apply one last skip connection between the input and output of the residual-block loop

        up_sampling = layers.Add()([prelu_1, conv_2])

        # Upsample the data to the high-resolution dimensions

        for index in range(int(np.log2(self.us))):

            up_sampling = up_sampling_block(up_sampling, 3, 256, 1)

        # Apply the last convolutional layer to assert the number of filter is equal to the desired number of channels.

        outputs = layers.Conv2D(filters = self.channels, kernel_size = 9, strides = 1, padding = "same", data_format='channels_last')(up_sampling)

        # Connect input and output layers

        generator = keras.Model(inputs, outputs, name='SRGAN-Generator')

        """
            Discriminator model
        """

        def discriminator_block(model, filters, kernel_size, strides):
            """
                Function to generate discriminator blocks.
            
            :param model:       Tensorflow tensor containing the internal model state 
            :param filters:     Integer containing the number of filters to apply in the convolution operation.
            :param kernel_size: Integer containing the kernel (or filter) size for the convolutional operations
            :param strides:     Integer containing the stride value to ba applied in the convolutional operations.
            :return model:      Tensorflow tensor
            """

            # Apply convolutional operation
                
            model = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_last')(model)

            # Apply Leaky ReLU activation function

            model = layers.LeakyReLU(alpha = 0.2)(model)
            
            return model

        # Define input layer

        inputs = keras.Input(shape=(self.ny*self.us, self.nx*self.us, self.channels), name='high-res-input')

        # Apply a convolutional layer
        
        model = layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same", data_format='channels_last')(inputs)

        # Apply a Leaky ReLU activation function

        model = layers.LeakyReLU(alpha = 0.2)(model)

        # Apply 7 discriminator blocks 

        model = discriminator_block(model, 64, 3, 4)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)

        # Flatten the tensor into a vector

        model = layers.Flatten()(model)

        # Apply a fully-conncted layer 

        model = layers.Dense(1024)(model)

        # Apply a convolutional layer

        model = layers.LeakyReLU(alpha = 0.2)(model)

        # Apply a fully-conncted layer 

        model = layers.Dense(1)(model)

        # Apply a sigmoid connection function

        model = layers.Activation('sigmoid')(model) 

        # Connect input and output layers
        
        discriminator = keras.Model(inputs=inputs, outputs = model, name='SRGAN-Discriminator')

        """
            Generator loss
        """

        # Define generator loss as a function to be returned for its later use during the training

        def custom_mse_loss(y_pred, y_true, flag):
            """
                Custom function to compute the mean-squared error between target and predicted data only for a certain number of bins.
            
            :param y_pred: Tensorflow tensor containing the predicted high-resolution fields
            :param y_true: Tensorflow tensor containing the target high-resolution fields
            :param flag:   Tensorflow tensor containing boolean information regarding the bins in the target data that have information
            :return loss:  Float containg the mean-squared error
            """

            # Compute the number of bins in the target high-resolution data that contains information

            N = tf.reduce_sum(flag)

            # Compute the conditional mean-squared error

            loss = tf.math.divide(
                tf.reduce_sum(
                tf.math.square(
                    tf.math.subtract(
                    y_true,
                    y_pred
                    )
                )
                ), 
                N
            )

            return loss

        def generator_loss(fake_Y, hr_predic, hr_target, fl_target):
            """
                Function to compute the generator loss as the mean-squared error between the target and the predicted high-resolution fields plus and adversarial error for the perdicted fields.

            :param fake_Y:    Tensorflow vector of dimensions batch size x 1 containing the labels of the predicted data.
            :param hr_predic: Tensorflow tensor containing the predicted high-resolution fields
            :param hr_target: Tensorflow tensor containing the target high-resolution fields
            :param fl_target: Tensorflow tensor containing the information about which bins in the high-resolution target data contain information.
            :return loss:     Float containing the generator loss
            """

            # Define binary cross-entropy function

            cross_entropy = tf.keras.losses.BinaryCrossentropy()

            # Compute the capability of the generator to cause the discriminator to misidentify the predicted data as real, adding a small perturbation for stability issues

            adversarial_loss = cross_entropy(
                np.ones(fake_Y.shape) - np.random.random_sample(fake_Y.shape) * 0.2, 
                fake_Y
            )

            # Compute the mean-squared error between the target and predicted data only as a function of the bins in the target data that contain information

            content_loss = custom_mse_loss(
                hr_target, 
                tf.math.multiply(hr_predic, fl_target), 
                fl_target
            )

            # Compute loss

            loss = content_loss + 1e-3*adversarial_loss

            return loss

        """
            Discriminator loss
        """

        # Define discriminator loss as a function to be returned for its later use during the training

        def discriminator_loss(real_Y, fake_Y):
            """
                Function to compute the discriminator loss as the mean value of the binary cross-entropy for the target and predicted labels.

            :param real_Y:      Tensorflow vector of dimensions batch size x 1 containing the labels of the target data.
            :param fake_Y:      Tensorflow vector of dimensions batch size x 1 containing the labels of the predicted data.
            :return total_loss: Float containing the mean value of the binary cross-entropy for the target and predicted labels.
            """

            # Define binary cross-entropy function

            cross_entropy = tf.keras.losses.BinaryCrossentropy()

            # Compute the capability of the discriminator to identify the target data as real, adding a small perturbation for stability issues

            real_loss = cross_entropy(np.ones(real_Y.shape) - np.random.random_sample(real_Y.shape)*0.2, real_Y)

            # Compute the capability of the discriminator to identify the predicted data as fake, adding a small perturbation for stability issues

            fake_loss = cross_entropy(np.random.random_sample(fake_Y.shape)*0.2, fake_Y)

            # Compute mean value

            total_loss = 0.5 * (real_loss + fake_loss)
            
            return total_loss

        return generator, discriminator, generator_loss, discriminator_loss


    def optimizer(self, learning_rate):

        generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

        return generator_optimizer, discriminator_optimizer



