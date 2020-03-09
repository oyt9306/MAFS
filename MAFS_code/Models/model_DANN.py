# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 22:54:23 2019

@author: YeongTakOh
"""
import random
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.python.keras import utils
import tensorflow.keras.optimizers as ko
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import matplotlib.pyplot as plt
import pickle as pk
from tensorflow.keras.models import model_from_json 
from sklearn.metrics import accuracy_score
from Gradient_Reverse_Layer import *

#%%
# This code is for deep-supervised domain adaptation for motor fault detecction
mypath_origin = "D:\Google Drive\대학원\석사\Personal Files\Python Code\Research_1_SDA"
     
   
def feature_extractor_cnn(input_sig):
    # First Layer
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
    output = kl.GlobalAveragePooling1D()(x)
    return output

def feature_extractor_res(input_sig):
    # First Layer
    conv0 = kl.Conv1D(64, kernel_size=128, padding="same")(input_sig)
    acti0 = kl.Activation("relu")(conv0)
    pool0 = kl.MaxPooling1D(pool_size=2, padding="same", strides=2)(acti0)
    # Block 1
    input1 = pool0
    BN1    = kl.BatchNormalization(momentum=0.8)(input1)
    acti1  = kl.Activation("relu")(BN1)
    conv1  = kl.Conv1D(64, kernel_size=6, padding="same")(acti1)
    pool1  = kl.MaxPooling1D(pool_size=2, padding="same", strides=2)(kl.add([input1,conv1]))
    acti2  = kl.Activation("relu")(pool1)
    # Block 2
    input2 = acti2
    BN2    = kl.BatchNormalization(momentum=0.8)(input2)
    acti3  = kl.Activation("relu")(BN2)
    conv2  = kl.Conv1D(64, kernel_size=4, padding="same")(acti3)
    pool2  = kl.MaxPooling1D(pool_size=2, padding="same", strides=2)(kl.add([input2,conv2]))
    acti4  = kl.Activation("relu")(pool2)
    # Block 3
    input3 = acti4
    BN3    = kl.BatchNormalization(momentum=0.8)(input3)
    acti5  = kl.Activation("relu")(BN3)
    conv3  = kl.Conv1D(64, kernel_size=4, padding="same")(acti5)
    pool3  = kl.MaxPooling1D(pool_size=2, padding="same", strides=2)(kl.add([input3,conv3]))
    output = kl.GlobalAveragePooling1D()(pool3)
    return output

def classifier(inp):
    ''' 
    This function defines the structure of the classifier part.
    '''
    out = kl.Dense(128, activation="relu")(inp)
    out = kl.Dense(32, activation="relu")(out)
    classifier_output = kl.Dense(2, activation="sigmoid", name="classifier_output")(out)
    return classifier_output

def discriminator(inp):
    ''' 
    This function defines the structure of the discriminator part.
    '''
    out = kl.Dense(128, activation="relu")(inp)
    discriminator_output = kl.Dense(2, activation="sigmoid", name="discriminator_output")(out)
    return discriminator_output

def _build(batch_size):
    '''
    This function builds the network based on the Feature Extractor, Classifier and Discriminator parts.
    '''
    L_window = 3600
    inp = kl.Input(shape=(L_window,1), name="main_input")
    feature_output = feature_extractor_res(inp)
    feature_output_grl = GradientReversal(feature_output)
    
    labeled_feature_output = kl.Lambda(lambda x: K.switch(K.learning_phase(), 
                                                          K.concatenate([x[:int(batch_size//2)],
                                                                         x[:int(batch_size//2)]], axis=0), x), 
                                       output_shape=lambda x: x[0:])(feature_output_grl)
    classifier_output = classifier(labeled_feature_output)
    discriminator_output = discriminator(feature_output)
    model = Model(inputs=inp, outputs=[discriminator_output, classifier_output])
    model.summary()
    return model

def batch_generator(trainX, trainY=None, batch_size=1, shuffle=True):
    '''
    This function generates batches for the training purposes.
    '''
    L_window = 3600
    if shuffle:
        index = np.random.randint(0, len(trainX) - batch_size)
    else:
        index = np.arange(0, len(trainX), batch_size)
    while trainX.shape[0] > index + batch_size:
        batch_signals = trainX[index : index + batch_size]
        batch_signals = batch_signals.reshape(batch_size, L_window, 1)
        if trainY is not None:
            batch_labels = trainY[index : index + batch_size]
            yield batch_signals, batch_labels
        else:
            yield batch_signals
        index += batch_size
        
#%%
def train_DANN(trainX, trainDX, testX, testDX, trainY, trainDY, testY, testDY, epochs=1, batch_size=1, verbose=True, save_model=None):
    '''
    This function trains the model using the input and target data, and saves the model if specified.
    '''
    model = _build(batch_size)
    for cnt in range(epochs): 
        p = np.float(cnt) / epochs
        lr = 0.01 / (1. + 10 * p)**0.75
        # Prepare batch data for the model training.
        Labeled   = batch_generator(trainX, trainY, batch_size=batch_size // 2)
        UNLabeled = batch_generator(trainDX, batch_size=batch_size // 2)
        
        model.compile(optimizer=Adam(lr), 
              loss={'classifier_output': 'binary_crossentropy', 
                    'discriminator_output': 'binary_crossentropy'},
              loss_weights={'classifier_output': 0.5, 
                            'discriminator_output': 0.5})


        for batchX, batchY in Labeled:
            try:
                batchDX = next(UNLabeled)
            except:
                UNLabeled = batch_generator(trainDX, batch_size=batch_size // 2)
                    
            # Get the batch for unlabeled data. If the batches are finished, regenerate the batches agian.
            batchDX = UNLabeled
            # Combine the labeled and unlabeled images along with the discriminative results.
            combined_batchX = np.concatenate((batchX, batchDX))
            batch2Y = np.concatenate((batchY, batchY))
            combined_batchY = np.concatenate((np.tile([0, 1], [batchX.shape[0], 1]), np.tile([1, 0], [batchDX.shape[0], 1])))
            # Train the model
            metrics = model.train_on_batch({'main_input': combined_batchX}, 
                                                {'classifier_output': batch2Y, 'discriminator_output':combined_batchY})
        
        print("Epoch {}/{}\t[Source loss: {:.4}, Target loss: {:.4}, Domain loss: {:.4}]".format(cnt+1, 
                                                                                                 epochs,
                                                                                                 metrics[0], 
                                                                                                 metrics[1], 
                                                                                                 metrics[2]))       
        trainY_hat = model.predict(trainX).argmax(1)
        trainDY_hat = model.predict(trainDX).argmax(1)
        testY_hat = model.predict(testX).argmax(1)
        testDY_hat = model.predict(testDX).argmax(1)

        p1 = accuracy_score(trainY, trainY_hat)
        p2 = accuracy_score(trainDY, trainDY_hat)
        p3 = accuracy_score(testY, testDY_hat)
        p4 = accuracy_score(testDY, testY_hat)

        print("Source) Acc of classifier and discriminator: [{:.3}, {:.3}]".format(p1, p2))
        print("Target) Acc of classifier and discriminator: [{:.3}, {:.3}]".format(p3, p4))
    model.save_weights("DANN_model.h5")
    return model
        
#%%
def load_model():
    # Load the model
    print("Load model from pre-processing") 
    json_file = open("DANN_model.json","r") 
    loaded_model_json = json_file.read() 
    json_file.close() 
    loaded_model = model_from_json(loaded_model_json)
    # Load the weight
    print("Load weight from pre-processing") 
    loaded_model.load_weights("DANN_model.h5") 
    model = loaded_model
    print('Done \n') 
    return model 

def evaluate(model, testX, testY):
    # This function evaluates the model, and generates the predicted classes.
    acc = model.evaluate(testX, testY)
    print("The classifier and discriminator metrics for evaluation are [{}, {}]".format(acc[0], acc[1]))