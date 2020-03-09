# -*- coding: utf-8 -*-
"""
This code contains, 
main_func of 1) pre-training, 2) training Binary-supervised domain adaptation

Created on Wed Jun 19 11:31:01 2019
Ver1) Modified on Mon May 3 12:18:30 2020

@author: YeongTakOh
"""
import numpy as np
import os
from Models.model_train import *
from Models.model_load import *
from Models.model_plot import *
from Models.model_SDA import *
from Models.Semi_SGAN import *
import tensorflow.keras.optimizers as ko
import matplotlib.pyplot as plt

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
mypath_load   = ml.load_mypath(N_load)
n_axis = 3 # 4-th joint

load_mode     = True
train_mode    = True
BSDA_mode     = True
overlap_mode  = True
few_shot_mode = True
SSGAN_mode    = True
DANN_mode     = True

#%% Dataset generation
# TODO : Data loading
if load_mode == True:
    mypath1 = load_mypath(0) # Simple motion
    mypath2 = load_mypath(1) # Normal Welding
    mypath3 = load_mypath(2) # Fast Welding
    # Select the case
    n_case1 = 5
    n_case2 = 6
    n_case3 = 7
    case1 = case_selection(n_case1)
    case2 = case_selection(n_case2)
    case3 = case_selection(n_case3)
    # Select the Train/Val/Test number
    train_num1, val_num1  = train_val_split(case1)
    train_num2, val_num2, test_num2 = train_val_test_split(case2)
    train_num3, val_num3, test_num3 = train_val_test_split(case3)
    # Define the datasets
    X_tr1, y_tr1, X_val1, y_val1 = dataset_generation_mod(mypath1, train_num1, val_num1, L_window, L_shift)
    X_tr2, y_tr2, X_val2, y_val2, X_te2, y_te2 = dataset_generation(mypath2, train_num2, val_num2, test_num2, L_window, L_shift)
    X_tr3, y_tr3, X_val3, y_val3, X_te3, y_te3 = dataset_generation(mypath3, train_num3, val_num3, test_num3,  L_window, L_shift)
    # Generate the datasets
    X_train = X_tr1
    X_val   = X_val1
    y_train = y_tr1
    y_val   = y_val1
    # Select the axis
    X_train = axis_selection(X_train, n_axis)
    X_val   = axis_selection(X_val, n_axis)

if few_shot_mode == True:
    X_one, y_one = dataset_merging_code(n_axis, mypath3, L_window, L_shift, 3)
    X_two, y_two = dataset_merging_code(n_axis, mypath3, L_window, L_shift, 1)    

#%% Model Training
# TODO : Model build-up
train_selection = 0 # No training(0), Training(1), Fine-tunning(2)
backbone_name = 'Res_' # or, 'CNN_', 'Dense_'
backbone_network = network_selection(backbone_name, L_window)
backbone_network.summary()
backbone_network.compile(loss='binary_crossentropy', optimizer=ko.Adam(lr=1e-4), metrics=['accuracy'])
model_load = load_Pre_model(mypath_origin, backbone_name) # load the pre-model

if train_mode == True:
    if train_selection == 0:
        print('==== No training ====')        
    elif train_selection == 1: # Back-bone training
        if few_shot_mode == True:
            history = fit_model(backbone_name, backbone_network, X_one, y_one, X_two, y_two, BATCH_SIZE, EPOCHS)
            model_predictions = prediction_results_confusion(backbone_network, X_two, y_two)
        elif few_shot_mode == False:
            history = fit_model(backbone_name, backbone_network, X_train, y_train, X_val, y_val, BATCH_SIZE, EPOCHS)
            model_predictions = prediction_results_confusion(backbone_network, X_val, y_val)
            if backbone_name == 'Dense':
                history = fit_model(prediction_results_confusion, backbone_network, X_train.squeeze(), y_train, 
                                    X_val.squeeze(), y_val, BATCH_SIZE, EPOCHS)
                model_predictions = prediction_results_confusion(backbone_network, X_val.squeeze(), y_val)
        model_saving(backbone_name, backbone_network, n_case1)
        prediction_values(backbone_network, X_val, y_val, 'Before Source')
        t_SNE_result(X_val, y_val, L_window, backbone_network, -4)     
    elif train_selection == 2: # Transfer learning (with trained on the other domain)
        if few_shot_mode == True:
            backbone_network = freeze_model(model_load)
            history = fit_model(backbone_name, model_trans, X_one, y_one, X_two, y_two, BATCH_SIZE, EPOCHS)   
    # Show the results of prediction values for each-cycle
    data_N, data_F = ml.load_files(mypath_load)
    for i in range(len(data_N)):
        if i != N_want_target:
            X_temp, y_temp = dataset_merging_code(n_axis, mypath_load, L_window, L_shift, i)
            print('num: ', i)
            prediction_values(backbone_network, X_temp, y_temp, 'After SDA')
  
#%% BSDA(Binary-Supervised Domain Adaptation)      
# TODO : Domain adaptation task
SDA_train_mode = 1 # No training (0), BSDA traning(1)
model_SDA = model_load
if BSDA_mode == True:
    if SDA_train_mode == 0:
        print('==== No SDA training ====')
    elif SDA_train_mode == 1:
        # 1) Data-allocation
        X_1, y_1 = dataset_merging_code(n_axis, mypath_load, L_window, L_shift, N_want_target)
        X_2, y_2 = dataset_merging_code(n_axis, mypath_load, L_window, L_shift, N_want_test)
        # 2) Construct a Siamese network
        SDA_network = SDA_model_construction(model_SDA, L_window, alpha = 0.2)
        Acc, Test_Acc, Val_loss = fit_SDA_model(SDA_network, L_window , N_load, 
                                                X_train, X_1, X_2, 
                                                y_train, y_1, y_2)
        model_saving_SDA(model_SDA, model_name)
        print('Best accuracy for target sample data is {}.'.format(Acc))
        model_SDA = ml.load_SDA_model(mypath_origin, backbone_name)
        # Show the results of prediction values for each-cycle
        data_N, data_F = ml.load_files(mypath_load)
        for i in range(len(data_N)):
            if i != N_want_target:
                X_temp, y_temp = ml.dataset_merging_code(n_axis, mypath_load, L_window, L_shift, i)
                print('num: ', i)
                mp.prediction_values(SDA_network, X_temp, y_temp, 'After SDA')

#%% t-SNE with overlaped Region
model_SDA  = load_SDA_model(mypath_origin, backbone_name) # load the SDA model
if overlap_mode == True:
    X_target = []
    y_target = []
    for i in range(len(data_N)):
        if i != N_want_target:
            X_temp, y_temp = dataset_merging_code(n_axis, mypath_load, L_window, L_shift, i)
            X_target.append(X_temp)
            y_target.append(y_temp)
    X_target = np.array(X_target)
    y_target = np.array(y_target)
    shape_1 = X_target.shape
    shape_2 = y_target.shape
    X_target = X_target.reshape(shape_1[0]*shape_1[1],shape_1[2],1)
    y_target = y_target.reshape(shape_2[0]*shape_2[1],shape_2[2])
    X_overlap = np.vstack([X_val, X_target])
    y_overlap = np.vstack([y_val, y_target])
    t_SNE_result(X_overlap, y_overlap, L_window, model_load, -4)
    t_SNE_result(X_overlap, y_overlap, L_window, model_SDA, -4)

#%% SSGAN traning procedure
DX_tr = axis_selection(X_tr2, n_axis)
DX_val = axis_selection(X_val2, n_axis)
Dy_tr = y_tr2
Dy_val = y_val2

if SSGAN_mode == True:    
    # Main parameters 
    latent_dim = 350
    EPOCH = 100
    BATCH_SIZE = 64
    
    d_model, c_model = define_discriminator(L_window)
    g_model = define_generator(latent_dim)
    gan_model = define_gan(g_model, d_model)
    
    dataset = [DX_tr, Dy_tr]
    select_data = [DX_val, Dy_val]
    result_acc = train_SSGAN(g_model, d_model, c_model, gan_model, dataset, select_data, latent_dim, EPOCH, BATCH_SIZE)

#%% DANN training procedure
    if DANN_mode == True:
        dann_model = train_DANN(X_tr, DX_tr, X_val, DX_val, y_tr, Dy_tr, y_val, Dy_val, epochs=100, batch_size=64)
