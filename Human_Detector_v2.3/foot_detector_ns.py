# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:54:50 2021

@author: abhishek
"""
"""
Main script for Foot Detector subsystem to predict the location of foot in the
bboxes obtained from Human_Detector_v2. The output ndjson file contains bbox
coordinates along with foot keypoints and is stored in the output folder.

"""
# Importing the required modules
import ndjson
import os
import cv2
import sys
import math
import time
import click
import ntpath
import random
import numpy as np
from foot_detector_v2.MoveNet_core import MoveNet
from foot_detector_v2.draw_keypoints import keypoint_location
from foot_detector_v2.foot_keypoint_writer import frame_initialise, export_to_ndjson, generate_json, export_to_ndjson_with_scores

# CLI
@click.command()
@click.option('--file-path', help='Input video filepath')
@click.option('--total-frames', help='Number of frames to processs')
@click.option('--include-score', is_flag=True, help='Include confidence score in ouput. Default: False')
@click.option('--threshold', default=0.11, help='Custom position threshold. Default: 0.11')

def _start(file_path,total_frames,include_score,threshold):
    
    # Initialising the MoveNet class with the movenet object
    movenet = MoveNet()
    
    # Setting the file name
    file_name = ntpath.basename(file_path)
    file_name = os.path.splitext(file_name)[0]
    
    # Initialising frame capture
    video_capture = cv2.VideoCapture(file_path)
    height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # opening the ndjson file to read
    with open('cache/'+file_name+'-coordinates.ndjson') as json_file:
        data = ndjson.load(json_file)
    
    # Various parameters to keep a track of fps and current frame
    frame_no = 0
    pro_time = time.time()
    
    # Loop until every frame is read
    while frame_no < int(total_frames):
        
        # Reading the frame
        ret, frame = video_capture.read()
        
        # Break condition
        if(ret == False):
            if frame_no == 0:
                print("Foot Detector can't read video")
            break
        
        # Check if there is data available for the current frame
        try:
            
            # Prepare the writer for the current frame
            frame_initialise(frame_no)
            for box in data[frame_no]['people']:
                if(box[4]==1):
                    bbox = box
                    
                    # Obtain the cropped image inside the bbox
                    crop_img = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                    
                    # Send the cropped image to the predictor and obtain the output
                    keypoints = movenet.predictor(crop_img)
                    xl,yl,xr,yr,keypoint_score = keypoint_location(crop_img, keypoints, bbox[0], bbox[1], 0.3)
                    keypoint_score = keypoint_score.item()
                    
                    if(math.isnan(xl) and math.isnan(xr)):
                        x_out = xl
                        y_out = yr
                        keypoint_score = np.nan
                    elif math.isnan(xl):
                        x_out = xr
                        y_out = yr
                    elif math.isnan(xr):
                        x_out = xl
                        y_out = yl
                    else:
                        x_out = (xl+xr)/2
                        y_out = max(yr, yl)
                
                elif(box[4]==2):
                    x_out = np.nan
                    y_out = np.nan
                    keypoint_score = np.nan
                    
                else:
                    x_out = (box[0]+box[2])/2
                    y_out = (box[3]-(0.0138*height))
                    keypoint_score = random.uniform(0.9, 1.0)
                
                # Rounding scores to 2 decimal
                if not math.isnan(keypoint_score):
                    keypoint_score = round(keypoint_score,2)
                
                if include_score:
                    # Send output data to foot_keypoint_writer with scores
                    export_to_ndjson_with_scores(frame_no, int(box[0]), int(box[1]), int(box[2]), int(box[3]), x_out, y_out, box[5], keypoint_score)
                                                       
                else:
                    # Send output data to foot_keypoint_writer
                    export_to_ndjson(frame_no, int(box[0]), int(box[1]), int(box[2]), int(box[3]), x_out, y_out)

        
        # Data read error case        
        except IndexError:
            print("cache ndjson file is incomplete. Exiting",end='')
            print(".",end='')
            time.sleep(1)
            print(".",end='')
            time.sleep(1)
            print(".",end='')
            break
    
        # To get the progress   
        progress = ((frame_no+1)/int(total_frames))*100
        progress = round(progress,2)
        sys.stdout.write(f"\rProgress = {progress}%")
        sys.stdout.flush()
    
        frame_no+=1
    
    # Calling the generate_json to write data to new ndjson file
    try:
        generate_json(file_name,total_frames=frame_no,skp_scaling=True)
    
    except FileNotFoundError:
        os.mkdir("output/")
        generate_json(file_name,total_frames=frame_no,skp_scaling=True)
    
    print(" ")
    # Display processing time
    seconds = round(time.time()-pro_time)
    time2 = seconds
    hr = int(int(seconds)/3600)
    seconds = int(seconds - hr*3600)
    mins = int(int(seconds)/60)
    seconds = int(seconds - mins*60)
    print("Position estimation complete.... Processing time: "+"%02d" % (hr)+":"+"%02d" % (mins)+":"+"%02d" % (seconds))    
    
    # For total time taken
    with open("cache/t2.txt", "w") as time_file:
        time_file.write(str(time2))
    
    # Release the capture object
    video_capture.release()

# Takes command line input and passes it    
if __name__ == '__main__':
    _start()
    
# foot_detector ends here