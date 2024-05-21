# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:12:56 2021

@author: abhishek
"""
"""
This module will carry out the necessary scaling required to predict the
location of feet of detected humans. KNN imputation is combined with mean
imputation for best results.

"""
# Importing the required modules
import math
from sklearn.impute import KNNImputer

# Carry out knn imputation
def knn_Imputer (data):

    frame_no = 0
    
    X_train = []
    
    # Loop until every data is read
    while True:
        
        try:
            for bbox in data['frame no:'+str(frame_no)]:
                X_train.append(bbox)
    
        except KeyError:
            break
        
        frame_no+=1
    
    # Train the KNN Imputer
    imputer = KNNImputer()
    imputer.fit(X_train)
    
    # Fit the missing values using KNN Imputation
    Xnew = imputer.transform(X_train)
    
    # pointer to keep a track of index in Xnew
    ptr = 0
    for i in range(frame_no):
        for box in data['frame no:'+str(i)]:
            # If a data is missing do imputation
            if(math.isnan(box[4])):
                box[4] = Xnew[ptr][4]
                box[5] = Xnew[ptr][5]
            ptr+=1
    
    return data

# imputR ends here