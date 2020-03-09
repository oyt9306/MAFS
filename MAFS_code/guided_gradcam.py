# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:06:33 2019

@author: YeongTakOh
"""
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import cm
# tf.compat.v1.disable_eager_execution()

"""
Gradient Reversal Layer implementation for Keras
Credits:
https://github.com/michetonu/gradient_reversal_keras_tf/blob/master/flipGradientTF.py
"""

#%% 1D Guided Grad-CAM Code
def model_layers(model):
    for ilayer, layer in enumerate(model.layers):
        print("{:3.0f}, {:10}".format(ilayer, layer.name))

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def compile_saliency_function(model, act_num):
    input_sig = model.get_input_at(0)
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[act_num].output
    max_output = K.max(layer_output, axis=2)
    saliency = K.gradients(K.sum(max_output), input_sig)[0]
    return K.function([input_sig, K.learning_phase()], [saliency])


def modify_backprop(model, name):
    g = K.get_graph()
    with g.gradient_override_map({'ReLU': name}):
        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:] if hasattr(layer, 'activation')]
        # replace elu activation
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = tf.nn.relu
        # re-instanciate a new model
        new_model = model
    return new_model

def grad_cam_modify(model, layer_nm, x, num):
    """
    Args:
       model: model
       x: image input
       category_index: category index
       layer_name: last convolution layer name
    """
    # outputs = [layer.output for layer in model.layers]          # all layer outputs
    # class_output = model.output[:,num]
    class_output = model.get_output_at(num)
    convolution_output = model.get_layer(layer_nm).output # layer output
    grads = K.gradients(class_output, convolution_output)[0] # get gradients
    gradient_function = K.function([model.get_input_at(num)], [convolution_output, grads])  # get convolution output and gradients for input
    output, grads_val = gradient_function([x])
    output, grads_val = output[0], grads_val[0]
    weights = (1/grads_val.shape[0])*np.sum(grads_val, axis=0)
    cam = np.dot(output, weights)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    heatmap = heatmap - np.mean(heatmap)
    heatmap = cv2.resize(heatmap, (x.shape[0], x.shape[1]),cv2.INTER_CUBIC)
    return heatmap

# Plot with line-intensity
def plot_colourline(x,y,c, label_name):
    b = (c-np.min(c))/(np.max(c)-np.min(c))
    c_lst = [plt.cm.jet(a) for a in b]
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=c_lst[i], label = label_name)
    return

# Apply the Low-pass filter
def butter_lowpass(cutoff, nyq_freq, order):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def gradCAM_visual(x_grad1, y1, num):
    x_1 = np.squeeze(x_grad1, axis=1)
    y1  = y1.squeeze()
    Fs = 1000
    order = 4
    cutoff_frequency = 8
    # Low-pass filtering
    y_low1 = butter_lowpass_filter(y1, cutoff_frequency, Fs/2, order)
    # With Line-intensity
    n = 3600
    x = 1.*np.arange(n)
    plt.rcParams["figure.figsize"] = (6,4)
    plt.rcParams.update({'font.size': 12})
    y_1 = abs(y_low1)
    plot_colourline(x,x_1,y_1, 'Normal')
    plt.savefig('./GradCAM_'+str(num)+'.png', dpi=720)
