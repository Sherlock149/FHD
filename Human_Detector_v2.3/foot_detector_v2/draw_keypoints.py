# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 16:36:36 2021

@author: abhishek
"""
"""
This module is used for filtering keypoints, carry out frame transformation
and return the correct location of foot keypoints wrt the current frame.

"""
# Importing the required modules
import numpy as np

# Function to filter the required keypoints
def keypoint_selector(keypoints,height,width,threshold):

    # Extract the foot keypoints coordinates from the output data
    x_left = keypoints[0,0,15,1]
    y_left = keypoints[0,0,15,0]
    x_right = keypoints[0,0,16,1]
    y_right = keypoints[0,0,16,0]
    
    # Store the confidence scores corresponding to the keypoints
    cs_left = keypoints[0,0,15,2]
    cs_right = keypoints[0,0,15,2]
    
    # Convert coordinates wrt O' (the origin of the cropped image)
    x_left_absolute = width*x_left
    y_left_absolute = height*y_left
    x_right_absolute = width*x_right
    y_right_absolute = height*y_right
    
    # Remove low confidence keypoints
    if(cs_left > threshold):
        x_left_out = x_left_absolute
        y_left_out = y_left_absolute
    else:
        x_left_out = np.nan
        y_left_out = np.nan
    
    if(cs_right > threshold):
        x_right_out = x_right_absolute
        y_right_out = y_right_absolute
    else:
        x_right_out = np.nan
        y_right_out = np.nan
    
    cs_out = max(cs_left,cs_right)
    
    # Return the filtered keypoints with score
    return x_left_out, x_right_out, y_left_out, y_right_out, cs_out

# Function to perform frame transformation
def keypoint_location(image,keypoints,x1,y1,threshold):
    
    # Get the cropped image dimension
    height, width, _ = image.shape
    
    # Get filtered keypoints scaled acc to the cropped image
    xl,xr,yl,yr,cs = keypoint_selector(keypoints, height, width, threshold)
    
    # Obtain the coordinates wrt O (the origin of the current video frame)
    xl_out = x1 + xl
    xr_out = x1 + xr
    yl_out = y1 + yl
    yr_out = y1 + yr
    
    # Return the correct foot keypoint coordinates
    return xl_out,yl_out,xr_out,yr_out,cs

# draw_keypoints ends here