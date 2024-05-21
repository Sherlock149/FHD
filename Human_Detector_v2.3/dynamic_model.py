# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:24:23 2021

@author: abhishek
"""
"""
This scripts creates a model and trains it based on the detected data.
The model is saved on the cache and deleted after use.
Ensemble learning is applied using this model along with occlusion_detector.

"""
# Importing the required modules
import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def train_model():
    # Preparing the training data
    train_dataset = pd.read_csv('cache/Train_Data.csv')
    
    X_train = train_dataset.iloc[:,1]
    Y_train = train_dataset.iloc[:,2]
    
    # Model definition and training
    model = Sequential([
        
                layers.Dense(6, input_dim=1, activation='relu'),
                layers.Dense(6, activation='relu'),
                layers.Dense(1, activation='linear')
        
            ])
    
    model.compile(optimizer='adam',loss=('mse'))
    
    history = model.fit(X_train,Y_train,10,100,validation_split=0.1,verbose=0)
    
    # Store at cache for future use
    model.save('cache/dynamic_model.h5')
    os.remove('cache/Train_Data.csv')
    
# dynamic_model ends here