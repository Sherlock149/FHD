# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:41:17 2021

@author: abhishek
"""
"""
This is the module to write bbox coordinates and foot keypoints to ndjson file.
For k bounding boxes in a frame,
The data is stored as:
    frame no : [[tl1,br1,lf1,rf1,cs11,cs21],[tl2,br2,lf2,rf2,cs12,cs22],.....,[tlk,brk,lfk,rfk,cs1k,cs2k]]
Where:
    tl: Top Left coordinate, br: Bottom Right coordinate
    lf: Left Foot coordinate, rf: Right Foot coordinate
    cs1: Score of bbox, cs2: Score of foot keypoints

"""
# Importing the required modules
import ndjson
import json
import foot_detector_v2.imputR as imputR
from collections import OrderedDict

# Function to convert json data to ndjson format
def ndjson_creator(data,total_frames,outpt_file):
    with open('output/'+outpt_file + '-coordinates_with_keypoints.ndjson', 'w') as outfile:
        writer = ndjson.writer(outfile, ensure_ascii=False)
        
        # Creating one row for each frame
        for i in range(total_frames):
            dt = OrderedDict()
            dt["frame_no"] = i
            dt["people"] = []
            
            # Traversing through all the json data
            for box in data['frame no:'+str(i)]:
                try:
                    dt["people"].append([box[0],box[1],box[2],box[3],box[4],box[5],box[6],box[7]])
                    
                except IndexError:
                    dt["people"].append([box[0],box[1],box[2],box[3],box[4],box[5]])
                
            # Writing the row into the ndjson file
            writer.writerow(dt)
            
# Initialising an ordered dictionary
foot_data = OrderedDict()

# Create an empty list based on the frame_no
def frame_initialise(frame):
    foot_data['frame no:'+str(frame)] = []
    
# Insert coordinates into the specific frame_no
def export_to_ndjson (frame,x1,y1,x2,y2,x,y):
    foot_data['frame no:'+str(frame)].append([x1,y1,x2,y2,x,y])
    
# Insert coordinates with scores
def export_to_ndjson_with_scores (frame,x1,y1,x2,y2,x,y,cs1,cs2):
    foot_data['frame no:'+str(frame)].append([x1,y1,x2,y2,x,y,cs1,cs2])

# Write to ndjson file to output folder
def generate_json (file_name,total_frames,dat = foot_data,skp_scaling=False):
    outpt_filename = file_name
    
    if skp_scaling:
        foot_data = dat
    
    else:
        # Imputation for missing keypoints
        foot_data = imputR.knn_Imputer(dat)
    
    # Write the json data as ndjson file
    ndjson_creator(foot_data, total_frames, outpt_filename)

# foot_keypoint_writer ends here