# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:48:25 2021

@author: abhishek
"""
"""
This module will create the temporary dataset for the dynamic model in the cache

"""
# Importing the required modules
import pandas as pd

def create_dataset(data,h):
    X_train = []
    Y_train = []
    
    sufficient_data = True
    
    threshold = (135/1080)*h
    frame_no = 0
    while frame_no < 647:
        try:    
            for box in data['frame no:'+str(frame_no)]:
                height = box[3]-box[1]
                if(height >= threshold):
                    X_train.append(box[1]/1080.)
                    Y_train.append(height/1080.)
            frame_no += 1
    
        except KeyError:
            sufficient_data = False
            break
        
    df = pd.DataFrame(columns=(["X_train","Y_train"]))
    df["X_train"] = X_train
    df["Y_train"] = Y_train
    
    df.to_csv("cache/Train_Data.csv")
    
    return sufficient_data, len(X_train)

# dataset_generator ends here