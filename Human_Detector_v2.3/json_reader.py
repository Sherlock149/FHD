# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:10:13 2021

@author: abhishek
"""
"""
This module is for extracting ndjson data and visualize them

"""
# Importing the required modules
import ndjson
import cv2
import time
import math

# Setting the file path to the video
file_path = 'test_dataset/sample-videos_bpm-tokyo-2021-04_event20210423_conduta-001_20210423-194002.mp4'

# Initialising frame capture
video_capture = cv2.VideoCapture(file_path)

# opening the ndjson file to read
with open('output/sample-videos_bpm-tokyo-2021-04_event20210423_conduta-001_20210423-194002-coordinates_with_keypoints.ndjson') as json_file:
    data = ndjson.load(json_file)

# To keep a track of the current frame
frame_no = 0

# Loop until every frame is read
while True:
    
    # Reading the frame
    ret, frame = video_capture.read()
    
    # Break condition
    if(ret == False):
        print("Video has ended")
        break
    
    # Reading the coordinates of the bounding boxes corresponding to the current
    # frame from the ndjson file and draw them on the frame
    try:

        for box in data[frame_no]['people']:
            bbox = box
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            
            if(not math.isnan(bbox[4])):
                cv2.circle(frame, (int(bbox[4]),int(bbox[5])), radius=7, color=(0,0,255), thickness = -1)
            

    except IndexError:
        print("ndjson file is incomplete. Exiting",end='')
        print(".",end='')
        time.sleep(1)
        print(".",end='')
        time.sleep(1)
        print(".",end='')
        break
    
    # Show the frame with the bounding boxes
    cv2.imshow('preview', frame)

    # Saving 150 frames for manual accuracy analysis
    #if(frame_no >= 3000 & frame_no < 5900):
        #out_name = "Manual_accuracy_calculator/images/frame"+str(frame_no)+".jpg"
        #cv2.imwrite(out_name, frame)
 
    frame_no+=1
    
    # Quit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close all open windows
video_capture.release()
cv2.destroyAllWindows()

# json_reader ends here