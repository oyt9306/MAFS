# -*- coding: utf-8 -*-
"""
This code contains, 
back-up code for visualizing the results of the analysis

Created on Sun Sep 22 11:03:24 2019
Ver1) Modified on Mon May 3 12:18:30 2020

@author: YeongTakOh
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
import model_load as ml

mypath_origin = "D:\Google Drive\대학원\석사\Personal Files\Python Code\Research_1_SDA"

def prediction_values(model, X_test, Y_test, name):
    model_predictions = model.predict(X_test)
    Y_test_norm = np.zeros(len(Y_test))
    predictions_norm = np.zeros(len(Y_test))
    for i in range(len(Y_test)):
        Y_test_norm[i] = Y_test[i,0]
        predictions_norm[i] = np.round(model_predictions[i,0])
        score = accuracy_score(Y_test_norm, predictions_norm)
    plt.rcParams["figure.figsize"] = (3,3)
    plt.rcParams.update({'font.size': 12})
    plt.figure()
    plot_confusion_matrix(conf_mat=confusion_matrix(Y_test_norm, predictions_norm), colorbar=True, show_normed=True)
    print('==================================')
    print(str(name)+' Acc(%): ', 100*round(score,2))
    print('==================================')   

def prediction_results_plot(model, X_test, Y_test, L_window, L_shift):
    predictions = model.predict([X_test, X_test])
    N_len = len(Y_test)
    N_normal = np.count_nonzero(Y_test[:,0] == 1)
    N_fault = N_len - N_normal
    x_prob_1 = np.linspace(1, 0.001*(L_window+L_shift*N_normal), N_normal)
    x_prob_2 = np.linspace(1, 0.001*(L_window+L_shift*N_fault), N_fault)
    plt.rcParams.update({'font.size': 8})
    plt.figure()
    plt.figure(figsize=(5,3))
    plt.plot(x_prob_1,predictions[0][:N_normal,1], label='True Positive')
    plt.plot(x_prob_2,predictions[0][N_normal:,0], label='False Negative')
    plt.ylim([0, 1])
    plt.legend(loc='lower left')
    plt.show()
    return predictions

def pred_value_plot(n_axis, SDA_network, mypath_load, L_window, L_shift, N_load, N_target):
    os.chdir(mypath_origin+"./data/")
    data_N, data_F = ml.load_files(mypath_load)
    X_ex, y_ex = ml.dataset_merging(n_axis, mypath_load, L_window, L_shift, 1)
    L_size = len(X_ex)
    a = np.zeros((int(L_size/2), 2, len(data_N)))
    mat_mean_N = np.zeros((int(L_size/2),))
    mat_std_N  = np.zeros((int(L_size/2),))
    mat_mean_F = np.zeros((int(L_size/2),))
    mat_std_F  = np.zeros((int(L_size/2),))

    for i in range(len(data_N)):
        if i != N_target:
            X, y = ml.dataset_merging(n_axis, mypath_load, L_window, L_shift, i)
            print('Data types:', len(X))
            pred = prediction_results(SDA_network, X, y, L_window, L_shift)
            data_load = pred[0]
            a[:,0,i] = data_load[int(L_size/2):,0] #Normal
            a[:,1,i] = data_load[:int(L_size/2),1] #Fault
    print('Generation Procedures are done!')
    for j in range(int(L_size/2)):
        mat_mean_N[j] = np.sum(a[j,0,:])*(1/(len(data_N)-1))
        mat_std_N[j]  = np.std(a[j,0,:])
        mat_mean_F[j] = np.sum(a[j,1,:])*(1/(len(data_N)-1))
        mat_std_F[j]  = np.std(a[j,1,:])
    # Plotting Process    
    x = np.linspace(0,int(L_size/2),int(L_size/2))
    N_mul = 1
    os.chdir(mypath_origin)
    plt.rcParams["figure.figsize"] = (6,5)
    plt.rcParams.update({'font.size': 15})
    plt.figure() 
    if N_load == 1:
        plt.title('Welding')
    if N_load == 2:
        plt.title('Fast Welding')    
    plt.ylabel('Probability')
    plt.xlabel('Sliding window')
    plt.plot(x, mat_mean_N, 'b') # Normal 
    plt.fill_between(x, mat_mean_N - N_mul*mat_std_N, mat_mean_N + N_mul*mat_std_N, color='b', alpha=0.2)
    plt.plot(x, mat_mean_F, 'r') # Fault 
    plt.fill_between(x, mat_mean_F - N_mul*mat_std_F, mat_mean_F + N_mul*mat_std_F, color='r', alpha=0.2)
    plt.ylim(0, 1)
    plt.savefig('Predictions_'+str(N_load)+'_'+str(N_target)+'.png', dpi=360)