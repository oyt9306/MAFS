# -*- coding: utf-8 -*-
"""

This code contains, 
Semi-supervised GAN for generating the 1d-signal optimized for torque ripples

Created on Mon Feb 17 15:39:16 2020
Ver1) Modified on Mon May 3 12:18:30 2020

@author: YeongTakOh
"""

# example of semi-supervised gan for mnist
import os
import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
import matplotlib.pyplot as plt
from skimage.util import random_noise
from matplotlib import pyplot
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.datasets.mnist import load_data
import tensorflow.keras.models as km
import tensorflow.keras.optimizers as ko
from tensorflow.python.keras import backend as K

    
def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = kl.Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = kl.Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = kl.Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

# define the standalone supervised and unsupervised discriminator models
def define_discriminator(L_window, n_classes=2):
    input_sig = Input(shape=(L_window,1))
    # First block
    conv1  = kl.Conv1D(32, kernel_size=128, strides=1, padding='same')(input_sig)
    bn1    = kl.BatchNormalization(momentum=0.8)(conv1)
    activ1 = kl.LeakyReLU(0.2)(bn1)
    pool1  = kl.MaxPooling1D(2)(activ1)
    # Second block
    input1 = pool1
    bn2    = kl.BatchNormalization(momentum=0.8)(input1)
    activ2 = kl.LeakyReLU(0.2)(bn2)
    conv2  = kl.Conv1D(32, kernel_size=6, strides=1, padding="same")(kl.add([input1,activ2]))
    activ3 = kl.LeakyReLU(0.2)(conv2)
    pool2  = kl.MaxPooling1D(2)(activ3)
    # Third block
    input2 = pool2
    bn3    = kl.BatchNormalization(momentum=0.8)(input2)
    activ4 = kl.LeakyReLU(0.2)(bn3)
    conv3  = kl.Conv1D(32, kernel_size=6, strides=1, padding="same")(kl.add([input2,activ4]))
    activ5 = kl.LeakyReLU(0.2)(conv3)
    pool3  = kl.MaxPooling1D(2)(activ5)
    x = kl.GlobalAveragePooling1D()(pool3)
    x = kl.Dense(256, activation='relu')(x)
    x = kl.Dropout(0.3)(x)
    output_sig = kl.Dense(2, activation = 'sigmoid')(x)
    c_model = km.Model(input_sig, output_sig)
    c_model.compile(loss='binary_crossentropy', optimizer=ko.Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    # unsupervised output
    d_out_layer = kl.Lambda(custom_activation)(output_sig)
    # define and compile unsupervised discriminator model
    d_model = km.Model(input_sig, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=ko.Adam(lr=0.0002, beta_1=0.5))
    return d_model, c_model

# define the standalone generator model
def define_generator(latent_dim):
    input_sig = kl.Input(shape=(latent_dim,))
    n_nodes = 900 * 1
    gen = kl.Dense(n_nodes)(input_sig)
    gen = kl.LeakyReLU(alpha=0.2)(gen)
    gen = kl.Reshape((900, 1))(gen)
    # Start - block
    bn0    = kl.BatchNormalization(momentum=0.8)(gen)
    activ0 = kl.LeakyReLU(0.2)(bn0)
    conv0  = kl.Conv1D(128, kernel_size=16, padding="same")(kl.add[activ0,gen])
    # First block
    input1 = conv0
    bn1    = kl.BatchNormalization(momentum=0.8)(input1)
    activ1 = kl.LeakyReLU(0.2)(bn1)
    conv1  = kl.Conv1DTranspose(kl.add([input1,activ1]), filters=128, kernel_size=6, strides=2, padding='same')
    # Second block
    input1 = conv1
    bn2    = kl.BatchNormalization(momentum=0.8)(input1)
    activ2 = kl.LeakyReLU(0.2)(bn2)
    conv2  = kl.Conv1DTranspose(kl.add([input1,activ2]), filters=128, kernel_size=6, strides=2, padding='same')
    # Third block
    input2 = conv2
    bn3    = kl.BatchNormalization(momentum=0.8)(input2)
    activ4 = kl.LeakyReLU(0.2)(bn3)
    conv3  = kl.Conv1DTranspose(kl.add([input2,activ4]), filters=64, kernel_size=6, strides=1, padding='same')
    # Fourth block
    bn3    = kl.BatchNormalization(momentum=0.8)(conv3)
    activ5 = kl.LeakyReLU(0.2)(bn3)
    decoded = kl.Conv1D(1, kernel_size=4, strides=2, activation='linear', padding='same')(activ5) 
    print("shape of decoded {}".format(K.int_shape(decoded)))
    model = km.Model(input_sig, decoded)    
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	d_model.trainable = False
	gan_output = d_model(g_model.output)
	model = km.Model(g_model.input, gan_output)
	opt = ko.Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model


# select real samples
def generate_real_samples(dataset, n_samples):
	signals, labels = dataset
	ix = randint(0, signals.shape[0], n_samples)
	X, labels = signals[ix], labels[ix]
	y = ones((n_samples, 1))
	return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    sampled_noise = np.random.normal(0, 1, (latent_dim * n_samples))
    z_input = sampled_noise.reshape(n_samples, latent_dim)
    return z_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	z_input = generate_latent_points(latent_dim, n_samples)
	images = generator.predict(z_input)
	y = zeros((n_samples, 1))
	return images, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, c_model, latent_dim, dataset, n_samples=100):
    X_fake, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate the classifier model
    X_real, y = dataset
    _, acc = c_model.evaluate(X_real, y, verbose=0)
    print('Classifier Accuracy: %.3f%%' % (acc * 100))
    filename2 = 'g_model_%04d.h5' % (step+1)
    g_model.save(filename2)
    filename3 = 'c_model_%04d.h5' % (step+1)
    c_model.save(filename3)
    print('>Saved: %s, and %s' % (filename2, filename3))
    #plot
    plt.rcParams["figure.figsize"] = (8,4)
    plt.figure()
    plt.subplot(121)
    plt.plot(X_real[5,:])
    plt.subplot(122)
    plt.plot(X_fake[5,:])
    plt.savefig('./SGAN_result/Plot'+str(step+1)+'.png', dpi=720)
    return acc

# train the generator and discriminator
def train_SSGAN(g_model, d_model, c_model, gan_model, dataset, select_data, latent_dim, n_epochs, n_batch):
    X_sup, y_sup = select_data
    print(X_sup.shape, y_sup.shape)
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    n_steps = bat_per_epo * n_epochs
    half_batch = int(n_batch / 2)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    result_acc = []
    # manually enumerate epochs
    for i in range(n_steps):
        # update supervised discriminator (c)
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
        c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
        # update unsupervised discriminator (d)
        [X_real, _], y_real = generate_real_samples(dataset, half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update generator (g)
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # summarize loss on this batch
        if i%50==0:
            print('==> n_steps:%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
        # evaluate the model performance every so often
        if (i+1) % (bat_per_epo * 1) == 0:
            acc = summarize_performance(i, g_model, c_model, latent_dim, dataset)
            result_acc.append(acc)
    return result_acc