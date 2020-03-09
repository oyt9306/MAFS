# -*- coding: utf-8 -*-
"""
This code contains, 
execution of the 1d guided grad-CAM for visualization for highligted-regions

Created on Mon Mar  9 10:43:01 2020
Ver1) Modified on Mon May 3 12:18:30 2020

@author: YeongtakOh
"""

#%% 1D Grad-CAM algorithm
# TODO : Guided grad-CAM for trained model (Deep model Interpretation)
import os
import numpy as np
from model_load import *
from model_train import *
from guided_gradcam import *
import matplotlib.pyplot as plt
from Initialization import *
from scipy.signal import hilbert

np.random.seed(3)
# Data Number : [0] Axis-rotation, [1] Welding, [2] Fast Welding, [3] Spot Welding
# Hyperparameters
Fs_motor   = 1000
BATCH_SIZE = 128
EPOCHS     = 50
L_window   = 3600
L_shift    = 100
N_train    = 0
N_load     = 2
nb_classes = 2
alpha      = 0.35
N_want_target = 5
N_want_test   = 3
mypath_origin = "D:\Google Drive\대학원\석사\Personal Files\Python Code\Research_1_SDA"
mypath_saving = "D:\Research\Industrial Robot\데이터\Data"
mypath_load   = load_mypath(N_load)
n_axis = 3

# gCAM.model_layers(model_load)
model_name = 'Res_'
model_load = load_model(mypath_origin, model_name)
model_SDA  = load_SDA_model(mypath_origin, model_name)
X_grad, y_grad = dataset_merging_code(n_axis, mypath_load, L_window, L_shift, 0)

n_sel = 1
if n_sel == 0: # For, pre-trained model
    model_sel = model_load
    layer_heat = 'conv1d_6'
elif n_sel == 1: # For, domain-adapted model
    model_sel = model_SDA
    layer_heat = 'conv1d_32'

#%%
# Select the Signal number
n_sig = 20
x_grad1 = X_grad[n_sig,:] # Normal
x_grad1 = np.expand_dims(x_grad1, axis=0)
x_grad2 = X_grad[n_sig+int(len(X_grad)/2),:] # Fault
x_grad2 = np.expand_dims(x_grad2, axis=0)
print('Normal Prob: ', np.round(model_sel.predict(x_grad1),3))
print('Fault  Prob: ', np.round(model_sel.predict(x_grad2),3))
x_grad_want  = x_grad1

# 1) Grad-CAM
heat_map = grad_cam_modify(model_sel, layer_heat, x_grad_want, 0)

# 2) Guided grad-CAM
register_gradient()
guided_model = modify_backprop(model_sel, 'GuidedBackProp')
saliency_fn = compile_saliency_function(guided_model, layer_heat)
saliency = saliency_fn([x_grad_want, 0])
grad_cam_sig = saliency[0] * heat_map[np.newaxis]
grad_cam_sig = np.squeeze(grad_cam_sig, axis = 0)
guided_grad_cam = grad_cam_sig/np.max(grad_cam_sig)

# 3) Apply the Hilbert-Transfrom
x_grad_want = np.squeeze(x_grad_want, axis = 0)
analytic_signal = hilbert(guided_grad_cam)
amplitude_envelope = np.abs(analytic_signal)
gradCAM_visual(x_grad_want, amplitude_envelope, n_sig)
