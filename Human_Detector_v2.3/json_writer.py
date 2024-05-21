# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 16:47:47 2021

@author: abhishek
"""
"""
This is the module to write coordinate data to ndjson file.
For k bounding boxes in a frame,
The data is stored as:
    frame no : [[tl1,br1],[tl2,br2],.....,[tlk,brk]]
Where tl: Top Left coordinate, br: Bottom Right coordinate
If score is included, The data is stored as:
    frame no : [[tl1,br1,score1],[tl2,br2,score2],.....,[tlk,brk,scorek]]
It should be called by the human_detector script when running detections

"""
# Importing the required modules
import ndjson
import json
import occlusion_filter
import occlusion_filter_ns
from collections import OrderedDict

# Function to convert json data to ndjson format
def ndjson_creator(data,total_frames,outpt_file):
    with open('cache/'+outpt_file + '-coordinates.ndjson', 'w') as outfile:
        writer = ndjson.writer(outfile, ensure_ascii=False)
        # Creating one row for each frame
        for i in range(total_frames):
            dt = OrderedDict()
            dt["frame_no"] = i
            dt["people"] = []
                
            # Traversing through all the json data
            for box in data['frame no:'+str(i)]:
                try:
                    dt["people"].append([box[0],box[1],box[2],box[3],box[4],box[5]])
                    
                except IndexError:
                    dt["people"].append([box[0],box[1],box[2],box[3],box[4]])
                    
            # Writing the row into the ndjson file
            writer.writerow(dt)
    
# Initialising an ordered dictionary
data = OrderedDict()

# Create an empty list based on the frame_no
def frame_initialise(frame):
    data['frame no:'+str(frame)] = []
    
# Insert coordinates into the specific frame_no
def export_to_ndjson (frame,x1,y1,x2,y2):
    data['frame no:'+str(frame)].append([x1,y1,x2,y2,0])
    
# Insert coordinates with scores
def export_to_ndjson_with_scores (frame,x1,y1,x2,y2,cs):
    data['frame no:'+str(frame)].append([x1,y1,x2,y2,0,cs])

# Write to ndjson file to output folder
def generate_json (file_name,total_frames,h,dat = data,skp_scaling=False):
    outpt_filename = file_name
    
    if skp_scaling:
        data = occlusion_filter_ns.ns_occlusion_detector(dat, h)
    
    else:
        # Imputation for occluded bboxes
        data = occlusion_filter.occlusion_detector(dat, h)
    
    # Write the json data as ndjson file
    ndjson_creator(data, total_frames, outpt_filename)

# json_writer ends here