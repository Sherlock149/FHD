# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:29:11 2021

@author: abhishek
"""
"""
This script uses a trained ANN model to predict whether a given bbox is
occluded or not but does not enlarge the bbox (ns)

"""
# Importing the required modules
from tensorflow.keras.models import load_model

# Loading the ANN model
model = load_model('weights/occlusion_detector.h5')

# Function for prediction using the model
def ns_use_model(data,h):
    
    X = []
    Y = []
    
    frame_no = 0
    
    # Create test data (X)
    while True:
        try:
            for bbox in data['frame no:'+str(frame_no)]:
                X.append(bbox[1]/h)
            frame_no += 1
        
        except KeyError:
            break
    
    # Get predicted heights for each bbox
    Y = model.predict(X)
    
    return Y

# Function to detect whether an occlusion is due to a human
def ns_human_occlusion(data,frame, box):
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

def ns_occlusion_detector(data,h):
    
    # Get the predicted heights for the data
    heights = ns_use_model(data, h)
    
    frame_no = 0
   
    # pointer to iterate through the data
    ptr = 0
    
    # Loop till end of the data
    while True:
        try:
           
            # In case of occlusion, scale it and put a marker against it
            # The marker is needed for later steps
            for bbox in data['frame no:'+str(frame_no)]:
                if(heights[ptr]*h > (bbox[3]-bbox[1])):
                   
                    # If human occlusion mark it as nan
                    if(ns_human_occlusion(data,frame_no,bbox)):
                        bbox[4] = 2
                    
                    else:
                        bbox[4] = 1
                else:
                    pass
                
                ptr += 1
                
            frame_no += 1
        
        except KeyError:
            break
    
    # Return marked data
    return data
    
# occlusion_filter_ns ends here