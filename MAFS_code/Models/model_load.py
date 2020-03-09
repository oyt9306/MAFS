# -*- coding: utf-8 -*-

"""
This code contains, 
1) data load, 2) model load, 3) visualization parts 

Created on Mon May 13 15:18:30 2019
Ver1) Modified on Mon May 3 11:18:30 2020

@author: YeongTakOh
"""

import os
from os.path import isfile, join
import numpy as np
import scipy.io as sio
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mlxtend.plotting import plot_confusion_matrix
from model_train import *

np.random.seed(3)
number_of_classes = 2 #Total number of classes

#%% Load and process the datasets
# Concatenating the datasets
def generate_dataset_code(n_axis, mypath, L_window, L_shift, N_data, n_idx):
    """
    Datasets are constructed with '.mat' file after pre-processing (e.g. AN0001.mat, and AF0001.mat)
    """
    data_read_1, data_read_2 = load_files(mypath)
    X_concat = np.zeros((0,L_window,1))
    if n_idx == 0:
        X_data = sio.loadmat(mypath + data_read_1[N_data])['motor_normal'][:,:]  
        X_data = X_data[:,n_axis]
        X_data = np.expand_dims(X_data, axis = 1)
    elif n_idx == 1:
        X_data = sio.loadmat(mypath + data_read_2[N_data])['motor_fault'][:,:]
        X_data = X_data[:,n_axis]
        X_data = np.expand_dims(X_data, axis = 1)
    Num_window = round((len(X_data)-L_window)/L_shift)
    X1_data    = np.zeros((Num_window,L_window,1))
    for i in range(Num_window):
        X1_data[i,:] = X_data[L_shift*i:L_window+L_shift*i]
    X_concat = np.vstack((X_concat, X1_data)) 
    return X_concat

# Load lables of the files (with Binary-Categirical Input Shape)
def load_files_label(mypath, X_data1, N_num):
    number_of_classes = 2
    Data_num1, L_window, temp = X_data1.shape
    Label_set1 = np.zeros((Data_num1, number_of_classes))
    # Normal : [1, 0], Fault : [0, 1]
    if N_num == 0:
        Label_set1[:,0] = 1
        Label_set1[:,1] = 0
    elif N_num == 1:
        Label_set1[:,0] = 0
        Label_set1[:,1] = 1
    return Label_set1

# Merging the datasets (To construct normal and fault data)
def dataset_merging_code(n_axis, mypath, L_window, L_shift, N_data):
    X_Normal = generate_dataset_code(n_axis, mypath, L_window, L_shift, N_data, 0)
    X_Fault  = generate_dataset_code(n_axis, mypath, L_window, L_shift, N_data, 1)
    y_Normal = load_files_label(mypath, X_Normal, 0)
    y_Fault  = load_files_label(mypath, X_Fault, 1)
    X_merged = np.vstack((X_Normal, X_Fault))
    y_merged = np.vstack((y_Normal, y_Fault))
    return X_merged, y_merged

# Load the pre-trained model
def load_Pre_model(mypath, name):
    os.chdir(mypath+'./data')
    # Load the model
    print("Load pre model from pre-processing") 
    json_file = open("Source_model_"+name+".json","r") 
    loaded_model_json = json_file.read() 
    json_file.close() 
    loaded_model = km.model_from_json(loaded_model_json)
    # Load the weight
    print("Load pre weight from pre-processing") 
    loaded_model.load_weights("Source_model_"+name+".h5") 
    model = loaded_model
    print('Done \n') 
    return model 

# Load the domain-adapted model    
def load_SDA_model(mypath, name):
    os.chdir(mypath+'./data')
    print("Load adapted model from pre-processing") 
    json_file = open("SDA_total_model_"+name+".json", "r") 
    loaded_model_json = json_file.read() 
    json_file.close() 
    loaded_model = km.model_from_json(loaded_model_json)
    # Load the weight
    print("Load adapted weight from pre-processing") 
    loaded_model.load_weights("SDA_total_weight_"+name+".h5") 
    model = loaded_model
    print('Done \n') 
    return model 

# Truncating the network after the GAP layer
def create_truncated_model(L_window, trained_model, layer_num):
    model = trained_model
    print('Output layer is, ', model.layers[layer_num].name)
    model.outputs = [model.layers[layer_num].output]
    # Add if needed
    # model.add(kl.GlobalAveragePooling1D())
    for i, layer in enumerate(model.layers):
        layer.set_weights(trained_model.layers[i].get_weights())
    model.compile(loss='binary_crossentropy', optimizer=ko.Adam(lr=0.0001), metrics=['accuracy'])
    return model

# Create the pairs of overlapped data
def dataset_overlap(X_source, y_source, X_target, y_target, L_window):
    N_normal_s = np.count_nonzero(y_source[:,0] == 1)
    N_target_s = np.count_nonzero(y_target[:,0] == 1)
    X_normal = np.vstack((X_source[:N_normal_s-1,:], X_target[:N_target_s-1,:]))
    X_fault  = np.vstack((X_source[N_normal_s:,:], X_target[N_target_s:,:]))
    X_overlap = np.vstack((X_normal, X_fault))
    y_normal = np.vstack((y_source[:N_normal_s-1,], y_target[:N_target_s-1,]))
    y_fault  = np.vstack((y_source[N_normal_s:,], y_target[N_target_s:,]))
    y_overlap = np.vstack((y_normal, y_fault))
    return X_overlap, y_overlap
    
#%% Visualizations of the embedding space
def t_SNE(X_input):
    tsne = TSNE(n_components=2, verbose = 1, perplexity=40 , n_iter=300) # 원래 40
    X_tsne= tsne.fit_transform(X_input)
    return X_tsne

def t_SNE_plot(X_data, y_data, num):
    plt.figure(figsize=(5,5))
    index = np.argmax(y_data, axis=1)
    if num == 0:
        name = '_Original'
    elif num == 1:
        name = '_Trained'
    plt.style.use(['default'])
    # If needed,
    # legend_plt = ['Normal', 'Fault']
    for cl in range(2):
        color_map = ['red', 'blue']
        indices = np.where(index==cl)
        indices = indices[0]
        plt.scatter(X_data[indices,0], X_data[indices, 1], c = color_map[cl], s = 5)
        # plt.legend(fontsize=10, loc = 'upper right')
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
    plt.savefig('t-SNE'+str(name)+'.png', dpi=360)
    
def t_SNE_result(X_load, y_load, L_window, model, layer_num):
    X_origin = np.squeeze(X_load, axis=2)
    X_origin_tsne = t_SNE(X_origin)
    t_SNE_plot(X_origin_tsne, y_load, 0)
    truncated_model = create_truncated_model(L_window, model, layer_num)
    X_cnn = truncated_model.predict(X_load)
    X_cnn_tsne = t_SNE(X_cnn)
    t_SNE_plot(X_cnn_tsne, y_load, 1)
    return X_cnn_tsne

def t_SNE_result_Model(X_load, y_load, L_window, model, layer_num):
    X_origin = np.squeeze(X_load, axis=2)
    X_origin_tsne = t_SNE(X_origin)
    t_SNE_plot(X_origin_tsne, y_load, 0)
    model.outputs = [model.layers[layer_num].output]
    for i, layer in enumerate(model.layers):
        layer.set_weights(model.layers[i].get_weights())
    model.compile(loss='binary_crossentropy', optimizer=ko.Adam(lr=0.0001), metrics=['accuracy'])    
    X_cnn = model.predict([X_load, X_load])
    X_cnn_tsne = t_SNE(X_cnn)
    t_SNE_plot(X_cnn_tsne, y_load, 1)
    return X_cnn_tsne

def freeze_model(base_model):
    Freeze_num = 9
    # model
    model = base_model
    # freeze layers
    for layer in model.layers[0:Freeze_num]:
        layer.trainable = False
    for layer in model.layers[Freeze_num+1:]:
       layer.trainable = True
    layer = base_model.output
    model = km.Model(inputs=base_model.input, outputs=layer)
    model.compile(optimizer=ko.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model