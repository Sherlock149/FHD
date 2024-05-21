# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:09:22 2021

@author: abhishek
"""
"""
This is the module to write coordinate data to ndjson file for images

"""
# Importing the required modules
import ndjson
import occlusion_filter
from collections import OrderedDict

# Function to convert json data to ndjson format
def ndjson_creator(data,total_frames,outpt_file):
    with open('output/'+outpt_file + '-coordinates.ndjson', 'w') as outfile:
        writer = ndjson.writer(outfile, ensure_ascii=False)
        
        # Creating one row for each frame
        for i in range(total_frames):
            dt = OrderedDict()
            dt["frame_no"] = i
            dt["people"] = []
            
            # Traversing through all the json data
            for box in data['frame no:'+str(i)]:
                try:
                    dt["people"].append([box[0],box[1],box[2],box[3],box[4]])
                    
                except IndexError:
                    dt["people"].append([box[0],box[1],box[2],box[3]])
                
            # Writing the row into the ndjson file
            writer.writerow(dt)
    
# Initialising an ordered dictionary
data = OrderedDict()

# Create an empty list based on the frame_no
def image_frame_initialise(frame):
    data['frame no:'+str(frame)] = []
    
# Insert coordinates into the specific frame_no
def image_export_to_ndjson (frame,x1,y1,x2,y2):
    data['frame no:'+str(frame)].append([x1,y1,x2,y2])
    
# Insert coordinates with scores
def image_export_to_ndjson_with_scores (frame,x1,y1,x2,y2,cs):
    data['frame no:'+str(frame)].append([x1,y1,x2,y2,cs])

# Write to ndjson file to output folder
def image_generate_json (file_name,total_frames,dat = data):
    outpt_filename = file_name
    
    # Write the json data as ndjson file
    ndjson_creator(data, total_frames, outpt_filename)

# image_writer ends here