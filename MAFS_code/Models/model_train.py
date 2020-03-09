# -*- coding: utf-8 -*-
"""
This code contains, 
backend code of 1) pre-training, 2) data selection, 3) network selection part

Created on Sat Jun  1 11:33:04 2019
Ver1) Modified on Mon May 3 12:18:30 2020

@author: YeongTakOh
"""
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
from tensorflow.python.keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import shuffle
from mlxtend.plotting import plot_confusion_matrix

#%%
mypath_origin = "D:\Google Drive\대학원\석사\Personal Files\Python Code\Research_1_SDA"

# Load the file
def load_mypath(num):
    if num == 0:
        mypath = 'D:/Research/Industrial Robot/데이터/Motor/1DCNN/Vel_Concat/' 
    elif num == 1:
        mypath = 'D:/Research/Industrial Robot/데이터/Motor/1DCNN/Welding_filter/' 
    elif num == 2: 
        mypath = 'D:/Research/Industrial Robot/데이터/Motor/1DCNN/Welding_fast_filter/' 
    elif num == 3: 
        mypath = 'D:/Research/Industrial Robot/데이터/Motor/1DCNN/Spot_filter/' 
    return mypath
 
# Load the Data
def load_files(mypath):
    onlyfiles_1 = [f for f in os.listdir(mypath) if (isfile(join(mypath, f)) and f[1] == 'N')]
    onlyfiles_2 = [f for f in os.listdir(mypath) if (isfile(join(mypath, f)) and f[1] == 'F')]
    data_read_1 = [f for f in onlyfiles_1 if f[7] == 'm']
    data_read_2 = [f for f in onlyfiles_2 if f[7] == 'm']
    return data_read_1, data_read_2

def data_augmentation(mypath, data_num, L_window, L_shift, state_num):
    data_read_1, data_read_2 = load_files(mypath)
    X_concat_1 = np.zeros((0,L_window,6))
    X_concat_2 = np.zeros((0,L_window,6))
    print('Data generation for, ', state_num)
    for j in data_num:
        X_data_1 = sio.loadmat(mypath + data_read_1[j])['motor_normal'][:,:]  
        X_data_2 = sio.loadmat(mypath + data_read_2[j])['motor_fault'][:,:]
        Num_window_1 = round((len(X_data_1)-L_window)/L_shift)
        Num_window_2 = round((len(X_data_2)-L_window)/L_shift)
        X2_data_1    = np.zeros((Num_window_1,L_window,6))
        X2_data_2    = np.zeros((Num_window_2,L_window,6))
        for i in range(Num_window_1):
            X2_data_1[i,:] = X_data_1[L_shift*i:L_window+L_shift*i]
        for i in range(Num_window_2):
            X2_data_2[i,:] = X_data_2[L_shift*i:L_window+L_shift*i]
        X_concat_1 = np.vstack((X_concat_1, X2_data_1))
        X_concat_2 = np.vstack((X_concat_2, X2_data_2))
    return X_concat_1, X_concat_2

def label_generation(X_data1, X_data2):
    number_of_classes = 2
    Data_num1, L_window, temp = X_data1.shape
    Data_num2, L_window, temp = X_data2.shape
    Label_set1 = np.zeros((Data_num1, number_of_classes))
    Label_set2 = np.zeros((Data_num2, number_of_classes))
    # Normal : [1, 0], Fault : [0, 1]
    Label_set1[:,0] = 1
    Label_set1[:,1] = 0
    Label_set2[:,0] = 0
    Label_set2[:,1] = 1
    return Label_set1, Label_set2

def dataset_merging_train(X_data_1, X_data_2, y_data_1, y_data_2):
    X_concat = np.vstack((X_data_1, X_data_2))
    y_concat = np.vstack((y_data_1, y_data_2))
    X_concat, y_concat = shuffle(X_concat, y_concat)
    return X_concat, y_concat

def dataset_generation(mypath, train_num, val_num, test_num, L_window, L_shift):
    X_tr_Normal, X_tr_Fault   = data_augmentation(mypath, train_num, L_window, L_shift, 'Train')
    X_val_Normal, X_val_Fault = data_augmentation(mypath, val_num, L_window, L_shift, 'Validation')
    X_te_Normal, X_te_Fault   = data_augmentation(mypath, test_num, L_window, L_shift, 'Test')
    y_tr_Normal, y_tr_Fault   = label_generation(X_tr_Normal, X_tr_Fault)
    y_val_Normal, y_val_Fault = label_generation(X_val_Normal, X_val_Fault)
    y_te_Normal, y_te_Fault   = label_generation(X_te_Normal, X_te_Fault)
    X_train, y_train =  dataset_merging_train(X_tr_Normal, X_tr_Fault, y_tr_Normal, y_tr_Fault)
    X_val, y_val     =  dataset_merging_train(X_val_Normal, X_val_Fault, y_val_Normal, y_val_Fault)
    X_test, y_test   =  dataset_merging_train(X_te_Normal, X_te_Fault, y_te_Normal, y_te_Fault)
    return X_train, y_train, X_val, y_val, X_test, y_test

def dataset_generation_mod(mypath, train_num, val_num, L_window, L_shift):
    X_tr_Normal, X_tr_Fault   = data_augmentation(mypath, train_num, L_window, L_shift, 'Train')
    X_val_Normal, X_val_Fault = data_augmentation(mypath, val_num, L_window, L_shift, 'Validation')
    y_tr_Normal, y_tr_Fault   = label_generation(X_tr_Normal, X_tr_Fault)
    y_val_Normal, y_val_Fault = label_generation(X_val_Normal, X_val_Fault)
    X_train, y_train =  dataset_merging_train(X_tr_Normal, X_tr_Fault, y_tr_Normal, y_tr_Fault)
    X_val, y_val     =  dataset_merging_train(X_val_Normal, X_val_Fault, y_val_Normal, y_val_Fault)
    return X_train, y_train, X_val, y_val
    
def train_val_test_split(case):
    train = list(map(int, case[0]))
    val   = list(map(int, case[1]))
    test  = list(map(int, case[2]))
    # Broadcasting
    a1 = np.ones_like(train)
    a2 = np.ones_like(val)
    a3 = np.ones_like(test)
    train = train - a1
    val   = val - a2
    test  = test - a3
    return train, val, test

def train_val_split(case):
    train = list(map(int, case[0]))
    val   = list(map(int, case[1]))
    # Broadcasting
    a1 = np.ones_like(train)
    a2 = np.ones_like(val)
    train = train - a1
    val   = val - a2
    return train, val

def case_selection(n_case):
    # 40, 60, 80 / 100
    case_0 = [['1','2','3','4','5','15','16','17','18','19','20','21','22','23','24'], 
              ['10','11','12','13','14'],
              ['6','7','8','9']]
    # 40, 60, 80 / 100
    case_1 = [['1','3','5','9','11','13','17','19','21'], 
              ['2','4','6','10','12','14','18','20','22'],
              ['7','8','15','16','23','24']]
    # 40, 60, 100 / 80
    case_2 = [['1','3','7','9','11','15','17','19','23'], 
              ['2','4','8','10','12','16','18','20','24'],
              ['5','6','13','14','21','22']]
    # 40, 80, 100 / 60
    case_3 = [['1','5','7','9','13','15','17','21','23'], 
              ['2','6','8','10','14','16','18','22','24'],
              ['3','4','11','12','19','20']]
    # 60, 80, 100 / 40
    case_4 = [['3','5','7','11','13','15','19','21','23'], 
              ['4','6','8','12','14','16','20','22','24'],
              ['1','2','9','10','17','18']]
    # Knowledge transfer (No test data)
    case_5 = [['1','3','5','7','9','11','13','15','17','19','21','23'], 
              ['2','4','6','8','10','12','14','16','18','20','22','24']]   
    # Welding Classifier
    case_6 = [['1','3','5','7'], 
              ['2','6'],
              ['4','8']]
    # Fast-Welding Classifier
    case_7 = [['1','3','5'], 
              ['2'],
              ['4','6']]
    # Welding - SGAN
    case_8 = [['1','2','3','4','6','7','8'], 
              ['5']]
    # Fast Welding - SGAN
    case_9 = [['1','2','3','5','6'], 
              ['4']]
    case = [case_0, case_1, case_2, case_3, case_4, case_5, case_6, case_7, case_8, case_9]
    return case[n_case]

def save_data(mypath, X_data, y_data, name):
    os.chdir(mypath)
    print('\n-----------')
    print('Data saving for X_data...')
    np.save('X_'+str(name)+'.npy', X_data)
    print('Data saving for y_data...')
    np.save('y_'+str(name)+'.npy', y_data)
    print('Done!')
    
def load_data(mypath, name):  
    os.chdir(mypath)
    X_data = np.load('X_'+str(name)+'.npy') 
    y_data = np.load('y_'+str(name)+'.npy') 
    print('Done!')
    return X_data, y_data

def axis_selection(X_data, n_axis):
    X_data = X_data[:,:,n_axis]
    X_data = np.expand_dims(X_data, axis = 2)
    return X_data
    
def residual_network(L_window):
    input_sig = kl.Input(shape=(L_window,1))   
    # First Layer
    conv0 = kl.Conv1D(64, kernel_size=128, padding="same")(input_sig)
    acti0 = kl.Activation("relu")(conv0)
    pool0 = kl.MaxPooling1D(pool_size=2, strides=2, padding="same")(acti0)
    # Block 1
    input1 = pool0
    BN1    = kl.BatchNormalization(momentum=0.8)(input1)
    acti1  = kl.Activation("relu")(BN1)
    conv1  = kl.Conv1D(64, kernel_size=6, padding="same")(acti1)
    pool1  = kl.MaxPooling1D(pool_size=2, strides=2, padding="same")(kl.add([input1,conv1]))
    acti2  = kl.Activation("relu")(pool1)
    # Block 2
    input2 = acti2
    BN2    = kl.BatchNormalization(momentum=0.8)(input2)
    acti3  = kl.Activation("relu")(BN2)
    conv2  = kl.Conv1D(64, kernel_size=4, padding="same")(acti3)
    pool2  = kl.MaxPooling1D(pool_size=2, strides=2, padding="same")(kl.add([input2,conv2]))
    acti4  = kl.Activation("relu")(pool2)
    # Block 3
    input3 = acti4
    BN3    = kl.BatchNormalization(momentum=0.8)(input3)
    acti5  = kl.Activation("relu")(BN3)
    conv3  = kl.Conv1D(64, kernel_size=4, padding="same")(acti5)
    pool3  = kl.MaxPooling1D(pool_size=2, strides=2, padding="same")(kl.add([input3,conv3]))
    x = kl.GlobalAveragePooling1D()(pool3)
    x = kl.Dense(128, activation='relu')(x)
    x = kl.Dropout(0.3)(x)
    output_sig = kl.Dense(2, activation = 'sigmoid')(x)
    model = km.Model(input_sig, output_sig)
    return model

def cnn_network(L_window):
    input_sig = kl.Input(shape=(L_window,1))
    x = kl.Conv1D(32, kernel_size=128, strides=1, padding='same')(input_sig)
    x = kl.BatchNormalization()(x)
    x = kl.Activation("relu")(x)
    x = kl.MaxPooling1D(2)(x)
    x = kl.Conv1D(64, kernel_size=4, strides=1, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation("relu")(x)
    x = kl.MaxPooling1D(2)(x)
    x = kl.Conv1D(128, kernel_size=4, strides=1, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation("relu")(x)
    x = kl.MaxPooling1D(2)(x)
    x = kl.Conv1D(128, kernel_size=4, strides=1, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation("relu")(x)
    x = kl.MaxPooling1D(2)(x)
    x = kl.GlobalAveragePooling1D()(x)
    x = kl.Dense(128, activation='relu')(x)
    x = kl.Dropout(0.3)(x)
    output_sig = kl.Dense(2, activation = 'sigmoid')(x)
    model = km.Model(input_sig, output_sig)
    return model

def dense_network(L_window):
    model = kl.Sequential()
    model.add(kl.Dense(256, input_shape=(L_window,)))
    model.add(kl.Activation('relu'))
    model.add(kl.Dense(256))
    model.add(kl.Activation('relu'))
    model.add(kl.Dense(256))
    model.add(kl.Activation('relu'))
    model.add(kl.Dense(2, activation = 'sigmoid'))
    return model

def network_selection(name, L_window):
    if name == 'Res_':
        model = residual_network(L_window)
    elif name == 'CNN_':
        model = cnn_network(L_window)
    elif name == 'Dense_':
        model = dense_network(L_window)
    return model

def fit_model(name, model, X_train, Y_train, X_val, y_val, BATCH_SIZE, EPOCHS):
    history =  model.fit(X_train, Y_train, 
                         batch_size = BATCH_SIZE,
                         epochs = EPOCHS, 
                         validation_data=(X_val, y_val),
                         verbose=1,
                         shuffle=True)
    plt.rcParams["figure.figsize"] = (12,4)
    plt.rcParams.update({'font.size': 18})
    plt.figure()
    plt.style.use(['default'])
    plt.subplot(121)
    plt.title('Loss Plot')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc='upper right')
    plt.subplot(122)
    plt.title('Acc Plot')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc='lower right')
    plt.savefig('Training_history.png', dpi=360)
    plt.show()
    np.save('Hist_val_loss_'+str(name)+'.npy', history.history['val_loss'])
    return history

def prediction_results_confusion(model, X_test, Y_test):
    model_predictions = model.predict(X_test)
    Y_test_norm = np.zeros(len(Y_test))
    predictions_norm = np.zeros(len(Y_test))
    for i in range(len(Y_test)):
        Y_test_norm[i] = Y_test[i,0]
        predictions_norm[i] = np.round(model_predictions[i,0])
        score = accuracy_score(Y_test_norm, predictions_norm)
    print('\n-------- Confusion Matrix --------')
    plt.rcParams["figure.figsize"] = (6,5)
    plt.rcParams.update({'font.size': 15})
    plt.figure()
    plot_confusion_matrix(conf_mat=confusion_matrix(Y_test_norm, predictions_norm), colorbar=True, show_normed=True)
    print("Prediction Accuracy: ", 100*round(score,2),'%')
    return model_predictions

def model_saving(name, model, case):
    os.chdir(mypath_origin+'./data')
    # Saving the Weight and CNN model
    print("\n -------- Saved weights to disk --------")
    model.save_weights("Source_model_"+name+".h5")
    print("\n -------- Saved model to disk --------")
    model_json = model.to_json()
    with open("Source_model_"+name+".json","w") as json_file :
        json_file.write(model_json)