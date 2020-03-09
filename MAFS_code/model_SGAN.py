# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 22:27:05 2019

@author: YeongTakOh
"""
import numpy as np
import os
import model_train_1D as mt
import model_load as ml
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

np.random.seed(3)
# Data Number : [0] Axis-rotation, [1] Welding, [2] Fast Welding, [3] Spot Welding
# Hyperparameters
Fs_motor   = 1000
BATCH_SIZE = 128
EPOCHS     = 50
L_window   = 3600
L_shift    = 100
N_train    = 0
N_load     = 1
nb_classes = 2
alpha      = 0.35
N_want_target = 5
N_want_test   = 3
n_axis = 3

mypath_origin = "D:\Google Drive\대학원\석사\Personal Files\Python Code\Research_1_SDA"
mypath_saving = "D:\Research\Industrial Robot\데이터\Data"
mypath_load   = ml.load_mypath(N_load)

#%% Heterogenous
mypath1 = mt.load_mypath(0)
mypath2 = mt.load_mypath(1)
mypath3 = mt.load_mypath(2)

n_case1 = 1
case1 = mt.case_study(n_case1)
n_case2 = 8
case2 = mt.case_study(n_case2)
n_case3 = 9
case3 = mt.case_study(n_case3)

train_num1, val_num1, test_num1 = mt.train_val_test_split(case1)
train_num2, val_num2  = mt.train_val_split(case2)
train_num3, val_num3  = mt.train_val_split(case3)

X_tr1, y_tr1, X_val1, y_val1, X_te1, y_te1 = mt.dataset_generation(mypath1, train_num1, 
                                                                   val_num1, test_num1, 
                                                                   L_window, L_shift)
X_tr2, y_tr2, X_val2, y_val2 = mt.dataset_generation_mod(mypath2, train_num2, 
                                                         val_num2,
                                                         L_window, L_shift)
X_tr3, y_tr3, X_val3, y_val3 = mt.dataset_generation_mod(mypath3, train_num3, 
                                                         val_num3, 
                                                         L_window, L_shift)

#%%X_train = mt.axis_selection(X_train, n_axis)
from Semi_SGAN_ver1 import *

# Target domain - Welding
DX_tr = mt.axis_selection(X_tr2, n_axis)
DX_val = mt.axis_selection(X_val2, n_axis)

Dy_tr = y_tr2
Dy_val = y_val2

# Main functions 
latent_dim = 350
EPOCH = 100
BATCH_SIZE = 64

d_model, c_model = define_discriminator(L_window)
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)

dataset = [DX_tr, Dy_tr]
select_data = [DX_val, Dy_val]
result_acc = train(g_model, d_model, c_model, gan_model, dataset, select_data, latent_dim, EPOCH, BATCH_SIZE)
plt.plot(result_acc)

# Model load and predictions
# Test data에 대해서 prediction 결과 저

# from keras.utils import plot_model
# import pydot
# import graphviz
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot, plot_model

# plot_model(gan_model) # Plot the model
# SVG(model_to_dot(gan_model, show_shapes=True).create(prog='dot', format='svg'))
# plot_model(gan_model, to_file = 'SGAN.png', show_shapes='True', show_layer_names='True')
