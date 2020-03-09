# -*- coding: utf-8 -*-
"""
This code is for deep binary-supervised domain adaptation for motor fault detecction

Created on Sun Jun  9 22:54:23 2019
Ver1) Modified on Mon May 3 12:18:30 2020

@author: YeongTakOh
"""
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import utils
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko

mypath_origin = "D:\Google Drive\대학원\석사\Personal Files\Python Code\Research_1_SDA"

def rand_sampling(X_data, y_data, N_num1, N_num2):
    N_normal = []
    N_fault  = []
    for i in range(len(y_data)):
        if y_data[i,0] == 1:
           N_normal.append(i)
        elif y_data[i,0] == 0:
           N_fault.append(i)
    Normal_sample = random.sample(N_normal, N_num1)
    Fault_sample  = random.sample(N_fault, N_num2)
    X_data_1 = []
    X_data_2 = []
    y_data_1 = []
    y_data_2 = []
    for i in range(len(Normal_sample)):
        X_data_1.append(X_data[Normal_sample[i],:,:])
        y_data_1.append(y_data[Normal_sample[i]])
    for i in range(len(Fault_sample)):
        X_data_2.append(X_data[Fault_sample[i],:,:])
        y_data_2.append(y_data[Fault_sample[i]])
    X_data_3 = np.vstack((X_data_1, X_data_2))
    y_data_3 = np.vstack((y_data_1, y_data_2))
    return X_data_3, y_data_3
   
def model_structure(model):
    new_model = km.Model(model.inputs, model.layers[-4].output)
    return new_model 

def euclidean_distance(vects):
    eps = 1e-3
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), eps))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def SDA_model_construction(pre_model, L_window, alpha):
    nb_classes = 2
    input_shape = (L_window, 1)
    input_source = kl.Input(shape=input_shape)
    input_target = kl.Input(shape=input_shape)
    processed_source = pre_model(input_source)
    processed_target = pre_model(input_target)
    out1 = kl.Dense(128)(processed_source)
    out1 = kl.Dropout(0.3)(out1)
    out1 = kl.Dense(nb_classes)(out1)
    out1 = kl.Activation('sigmoid', name='classification')(out1)
    distance = kl.Lambda(euclidean_distance, 
                         output_shape = eucl_dist_output_shape, 
                         name='CSA')([processed_source, processed_target])
    SDA = km.Model(inputs=[input_source, input_target], 
                   outputs=[out1, distance], name = 'SDA')
    Adam = ko.Adam(lr=0.0001)
    SDA.compile(loss={'classification': 'binary_crossentropy', 'CSA': contrastive_loss}, 
                optimizer= Adam, 
                loss_weights={'classification': 1 - alpha, 'CSA': alpha})
    return SDA

def fit_SDA_model(model, L_window ,N_num, X_train, X_1, X_2, y_train, y_1, y_2):
    # X_train, y_train   : Source domain data 
    # X_1, y_1, X_2, y_2 : Target domain data (Train/Val)
    # Sampling
    N_source_1 = 400
    N_source_2 = 400
    nb_classes = 2
    EPOCH      = 10
    BATCH_SIZE = 128
    N_mul = 0.2
    if N_num == 1:
        print('For Welding case, \n')
    elif N_num == 2:
        print('For Fast-Welding case, \n')
    print('Training the model_Epoch for, ' +str(EPOCH)+' epoch')
    nn = BATCH_SIZE
    best_Acc = 0
    Val_loss = np.zeros(EPOCH)
    Test_Acc = np.zeros(EPOCH)
    for e in range(EPOCH):
        t1 = time.time()
        X_trs, y_trs = rand_sampling(X_train, y_train, N_source_1, N_source_2)
        Training_P=[] # Positive-pair
        Training_N=[] # Negative-pair
        for trs in range(len(y_trs)):
            for trt in range(len(y_1)):
                if y_train[trs, 0]==y_1[trt, 0]:
                    Training_P.append([trs,trt])
                else:
                    Training_N.append([trs,trt])
        random.shuffle(Training_P)
        random.shuffle(Training_N)
        Pos_num = round(N_mul*len(Training_P))
        Neg_num = round(N_mul*len(Training_N))
        Training = Training_P[:Pos_num]+Training_N[:Neg_num]
        X1=np.zeros([len(Training),L_window,1])
        X2=np.zeros([len(Training),L_window,1])
        y1=np.zeros([len(Training)])
        y2=np.zeros([len(Training)])
        yc=np.zeros([len(Training)])
        for i in range(len(Training)):
            in1, in2=Training[i]
            X1[i,:] = X_train[in1,:]
            X2[i,:] = X_1[in2,:]
            y1[i]   = y_train[in1, 0]
            y2[i]   = y_1[in2, 0]
            if y1[i]==y2[i]:
                yc[i] = 1
        y1 = utils.to_categorical(y1, nb_classes)
        y2 = utils.to_categorical(y2, nb_classes)
        t2 = time.time()
        print('Preprocessing time : ', format(t2-t1, '.3f'), ' [sec]')  
        t3 = time.time()
        for i in range(round(len(y2)/nn)):
            loss_1 = model.train_on_batch([X1[i*nn:(i + 1)*nn,:,:], X2[i*nn:(i + 1)*nn,:,:]], 
                                          [y1[i*nn:(i+1)*nn,], yc[i*nn:(i+1)*nn,]])
            loss_2 = model.train_on_batch([X2[i*nn:(i + 1)*nn,:,:], X1[i*nn:(i + 1)*nn,:,:]], 
                                          [y2[i*nn:(i+1)*nn,], yc[i*nn:(i+1)*nn,]])
        predictions = model.predict([X_1, X_1])
        predictions1 = model.predict([X_2, X_2])
        Acc_v = np.argmax(predictions[0], axis=1) - y_1[:,0]
        Acc_v1 = np.argmax(predictions1[0], axis=1) - y_2[:,0]
        Acc = (len(Acc_v) - np.count_nonzero(Acc_v) + .0000001) / len(Acc_v)
        Acc1 = (len(Acc_v1) - np.count_nonzero(Acc_v1) + .0000001) / len(Acc_v1)
        Acc = round(Acc, 3)
        Acc1 = round(Acc1, 3)
        t4 = time.time()
        print('----------------'+' [ '+'Epoch '+str(e+1)+' ] '+'----------------')
        print('Loss 1 : ', format(loss_1[0], '.5f'), 'Loss 2 : ', format(loss_2[0], '.5f'))
        print('Var acc :', Acc, 'Test acc :', Acc1)
        print('Training time for 1 epoch : ', format(t4-t3, '.3f'), ' [sec]')
        print('-----------------------------------------------------------------')
        # loss, acc append 
        if best_Acc < Acc1:
            best_Acc = Acc1
        Val_loss[e] = format(loss_2[0], '.5f')
        Test_Acc[e] = Acc1
    # Saving and Plot
    print('Data saving...')
    np.save('Loss_'+str(N_num)+'.npy', Val_loss)
    np.save('Acc_'+str(N_num)+'.npy', Test_Acc)
    plt.figure()
    plt.rcParams["figure.figsize"] = (8,4)
    plt.figure()
    plt.style.use(['default'])
    plt.title('Plot')
    plt.plot(Val_loss)
    plt.plot(Test_Acc)
    plt.xlabel('Epoch')
    plt.legend(['Loss', 'Acc'], loc='upper right')
    print('----- Done! -----')
    return best_Acc, Test_Acc, Val_loss

def model_saving_SDA(model, name):
    # Saving the Weight and CNN model
    os.chdir(mypath_origin+'./data')
    print("\n -------- Saved weights to disk --------")
    model.save_weights("SDA_total_weight_"+name+".h5")
    print("\n -------- Saved model to disk --------")
    model_json = model.to_json()
    with open("SDA_total_model_"+name+".json","w") as json_file :
        json_file.write(model_json)  