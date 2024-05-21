# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:29:11 2021

@author: abhishek
"""
"""
This script deploys weighted avg. ensemble learning using a trained ANN model 
and a dynamically trained one to predict whether a given bbox is occluded or 
not. In case of occlusion, the type of occlusion is also decided
and small window is created below to check for feet done by other modules.
Weights w1 and w2 is based on the data obtained.

"""
# Importing the required modules
import os
import math
import numpy as np
from sklearn.impute import KNNImputer
from tensorflow.keras.models import load_model
from dataset_generator import create_dataset
from dynamic_model import train_model

# Loading the ANN model
model = load_model('weights/occlusion_detector.h5')

# Function for prediction using the model
def use_model(data,h,flag,w1):
    
    X = []
    Y1 = []
    Y2 = []
    w2 = 3638
    
    frame_no = 0
    
    # Create test data (X)
    while True:
        
        try:
            for bbox in data['frame no:'+str(frame_no)]:
                X.append(bbox[1]/h)
            frame_no += 1
        
        except KeyError:
            break
    
    # If there's sufficient data train another model
    if flag:
        print("Training Dynamic Model")
        train_model()
        model2 = load_model('cache/dynamic_model.h5')
        print("Dynamic model ready!")
        os.remove('cache/dynamic_model.h5')
        
    # Get predicted heights for each bbox
    Y1 = model.predict(X)
    
    # Weighted average
    if flag:
        Y2 = model2.predict(X)
        n = len(Y2)
        for i in range(n):
            Y1[i] = (w2/((3*w1)+w2))*(Y1[i]) + (3*w1/((3*w1)+w2))*(Y2[i])
    
    return Y1

# Function to detect whether an occlusion is due to a human
def human_occlusion(data,frame, box):
    bucket = []
    
    # Traverse through the data
    for bbox in data['frame no:'+str(frame)]:
        flag = 1
        for i in range(4):
            if(bbox[i]!=box[i]):
                flag = 0
        if flag == 0:
            bucket.append(bbox)
    
    occlusion_probability = 0.0
    
    # Check for human occlusion
    for bbox in bucket:
        if box[3] >= bbox[1] and box[3] < bbox[3]:
            if box[0] <= bbox[2] and box[0] >= bbox[0]:
                occlusion_probability += ((bbox[2]-box[0])/(box[2]-box[0]))
            elif box[2] <= bbox[2] and box[2] >= bbox[0]:
                occlusion_probability += ((box[2]-bbox[0])/(box[2]-box[0]))
    
    # If the total occlusion of a bbox by other bboxes > 95% return true
    if occlusion_probability >= 0.95:
        return True
    else:
        return False

def occlusion_detector(data,h):
    # Send the data for creating dataset
    flag, w1 = create_dataset(data, h)
    
    # Get the predicted heights for the data
    heights = use_model(data, h, flag, w1)
    if flag == False:
        os.remove('cache/Train_Data.csv')
    
    # KNN imputation for human occlusion
    imputer = KNNImputer()
    
    frame_no = 0
   
    # pointer to iterate through the data
    ptr = 0
    X = []
    X_train = []
    
    # Loop till end of the data
    while True:
        try:
           
            # In case of occlusion, scale it and put a marker against it
            # The marker is needed for later steps
            for bbox in data['frame no:'+str(frame_no)]:
                if(heights[ptr]*h > (bbox[3]-bbox[1])):
                   
                    # If human occlusion mark it as nan
                    if(human_occlusion(data,frame_no,bbox)):
                        bbox[3] = np.nan
                        X_train.append(bbox)                    
                    else:
                        X_train.append(bbox)
                        bbox[3] = bbox[1]+(heights[ptr][0]*h)
                        bbox[4] = 1
                else:
                    X_train.append(bbox)
                
                # All data stored in X
                X.append(bbox)
                ptr += 1
                
            frame_no += 1
        
        except KeyError:
            break
    
    # For all human occlusion, get bboxes using KNN
    imputer.fit(X_train)
    Xnew = imputer.transform(X)
    frame_no = 0
    ptr = 0
    while True:
        try:
            for bbox in data['frame no:'+str(frame_no)]:
                if(math.isnan(bbox[3])):
                    bbox[3] = Xnew[ptr][3]
                ptr += 1
                
            frame_no += 1
        
        except KeyError:
            break
    
    # Return the newly scaled data
    return data
    
# occlusion_filter ends here    