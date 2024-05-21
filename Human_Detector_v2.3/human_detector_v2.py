# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 08:00:17 2021

@author: abhishek
"""
"""
This is the script that needs to be run to detect humans in a video feed and
get bounding box coordinates as ndjson output. yolov4 is used with keras having
tensorflow backend. ndjson file is stored in the output folder.

"""
# Importing the required modules
import cv2
import ntpath
import os
import sys
import click
import warnings
import faster_nms
import numpy as np
from PIL import Image
from timeit import time
from yolov4_core import Detector
from decoder import Decoder
from json_writer import frame_initialise, export_to_ndjson, generate_json, export_to_ndjson_with_scores
from image_writer import image_frame_initialise,image_export_to_ndjson,image_export_to_ndjson_with_scores,image_generate_json

nms_max_overlap = 1.0
warnings.simplefilter("ignore")

# CLI
@click.command()
@click.option('--file-path', help='Input video filepath')
@click.option('--gpu-num', default=1, help='Number of GPUs. Default: 1')
@click.option('--include-score', is_flag=True, help='Include confidence score in ouput. Default: False')
@click.option('--threshold', default=0.4, help='Confidence Score threshold. Default: 0.4')
@click.option('--skip-scaling', is_flag=True, help='Skip scaling. Default: False')


# main function
def run_detector(file_path,gpu_num,include_score,threshold,skip_scaling):
    
    # Initialising the Detector class with the yolo object
    yolo = Detector(gpu_number=gpu_num, cs=threshold)
    
    click.echo("Starting Human Detector...")
    file_name1 = ntpath.basename(file_path)
    file_name = os.path.splitext(file_name1)[0]
    img_check = os.path.splitext(file_name1)[1]
    
    # If input is image
    if img_check == ".jpg" or img_check == ".png":
        
        while True:
            
            # Needed for the loop to work
            file_name1 = ntpath.basename(file_path)
            file_name = os.path.splitext(file_name1)[0]
            
            # Read the image
            image = cv2.imread(file_path)
            
            # Get the shape and do proper scaling
            img_height, img_width, _ = image.shape
            scale_x = img_width/608.
            scale_y = img_height/608
            
            # Preprocessing followed by similar detection
            frame = cv2.resize(image, (608,608))
                
            # Convert from BGR to RGB and put as a PIL image input to the model
            image = Image.fromarray(frame[...,::-1])
            boxes, cs, classes = Detector.detect_from_frame(yolo,image)
            
            # Decode the output to process useful info
            detections = [Decoder(bbox, confidence, clas) for bbox, confidence, clas in
                          zip(boxes, cs, classes)] 
                
            boxes = np.array([ptr.tlwh for ptr in detections])
            scores = np.array([ptr.confidence for ptr in detections])
                
            # Run non-maxima suppression.
            nms_idxs = faster_nms.nms(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in nms_idxs]
                
            # Send output data to json_writer
            image_frame_initialise(0)
            for j in detections:
                bbox = j.tlbr()
                score = round(j.confidence, 2)
                if include_score:
                    image_export_to_ndjson_with_scores(0, int(scale_x*bbox[0]), int(scale_y*bbox[1]), int(scale_x*bbox[2]), int(scale_y*bbox[3]), score)
                else:
                    image_export_to_ndjson(0, int(scale_x*bbox[0]), int(scale_y*bbox[1]), int(scale_x*bbox[2]), int(scale_y*bbox[3]))
                       
            # Calling the generate_json to write data to ndjson file
            try:
                image_generate_json(file_name,total_frames=1)
            
            except FileNotFoundError:
                os.mkdir("output/")
                image_generate_json(file_name,total_frames=1)
            
            print("Output for image generated...")
            
            # Check if user wants to predict another image
            recheck_flag = input("Load another image? (Y/N) ")
            
            if recheck_flag == 'N' or recheck_flag == 'n':
                exit(0)
            
            else:
                file_path = input("Enter File Path: ")
    
    # Image detection ends here
                    
    
    # Initialising frame capture if file is video
    video_capture = cv2.VideoCapture(file_path)
    
    # Various parameters to keep a track of fps and current frame
    pro_time = -1
    fps = 0.0
    frame_no = 0
    
    # To maintain the default video resolution (Changes dynamically)
    video_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale_x = video_width/608.
    scale_y = video_height/608
    
    # Loop until every frame is read
    while True:
        
        try:
        
            # Reading the frame
            ret, frame = video_capture.read()
            
            # Break cases
            if ret != True:
                if pro_time == -1:
                    print("Video Read Error")
                    break
                else:
                    seconds = round(time.time()-pro_time)
                    time1 = seconds
                    hr = int(int(seconds)/3600)
                    seconds = int(seconds - hr*3600)
                    mins = int(int(seconds)/60)
                    seconds = int(seconds - mins*60)
                    print("Video Ended.... Processing time:"+"%02d" % (hr)+":"+"%02d" % (mins)+":"+"%02d" % (seconds))
                    break
            
            # Preprocessing
            frame = cv2.resize(frame, (608,608))
            
            t1 = time.time()
            
            # Convert from BGR to RGB and put as a PIL image input to the model
            image = Image.fromarray(frame[...,::-1])
            boxes, cs, classes = Detector.detect_from_frame(yolo,image)
        
            # Decode the output to process useful info
            detections = [Decoder(bbox, confidence, clas) for bbox, confidence, clas in
                         zip(boxes, cs, classes)] 
            
            boxes = np.array([ptr.tlwh for ptr in detections])
            scores = np.array([ptr.confidence for ptr in detections])
            
            # Run non-maxima suppression.
            nms_idxs = faster_nms.nms(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in nms_idxs]
            
            # Send output data to json_writer
            frame_initialise(frame_no)
            for j in detections:
                bbox = j.tlbr()
                score = round(j.confidence, 2)
                if include_score:
                    export_to_ndjson_with_scores(frame_no, int(scale_x*bbox[0]), int(scale_y*bbox[1]), int(scale_x*bbox[2]), int(scale_y*bbox[3]), score)
                else:
                    export_to_ndjson(frame_no, int(scale_x*bbox[0]), int(scale_y*bbox[1]), int(scale_x*bbox[2]), int(scale_y*bbox[3]))
                   
            # Storing start time if its the first frame        
            if  frame_no == 0:
                pro_time = time.time()
        
            # To get the fps
            fps = (fps + (1./(time.time()-t1))) / 2
            fps = round(fps,2)
            sys.stdout.write(f"\rFPS = {fps}")
            sys.stdout.flush()
            frame_no+=1
        
        # Quit condition
        except KeyboardInterrupt:   
            print(" ")
            seconds = round(time.time()-pro_time)
            time1 = seconds
            hr = int(int(seconds)/3600)
            seconds = int(seconds - hr*3600)
            mins = int(int(seconds)/60)
            seconds = int(seconds - mins*60)
            print("Forced quit.... Video runtime: "+"%02d" % (hr)+":"+"%02d" % (mins)+":"+"%02d" % (seconds))
            break
   
    print(" ")
    
    # Post processing time
    post_time = time.time()
    
    # Calling the generate_json to write data to cache file
    try:
        os.mkdir("cache/")
        generate_json(file_name,total_frames=frame_no,h=video_height,skp_scaling=skip_scaling)
    
    except FileExistsError:
        os.remove('cache/'+file_name+'-coordinates.ndjson')
        os.remove('cache/frame_info.txt')
        try:
            os.remove('cache/t1.txt')
        except FileNotFoundError:
            pass
        try:
            os.remove('cache/t2.txt')
        except FileNotFoundError:
            pass
        os.rmdir("cache/")
        os.mkdir("cache/")
        generate_json(file_name,total_frames=frame_no,h=video_height,skp_scaling=skip_scaling)
    
    seconds = round(time.time()-post_time)
    hr = int(int(seconds)/3600)
    seconds = int(seconds - hr*3600)
    mins = int(int(seconds)/60)
    seconds = int(seconds - mins*60)
    print("Post processing time: "+"%02d" % (hr)+":"+"%02d" % (mins)+":"+"%02d" % (seconds))
    
    # To be used for scaler
    with open("cache/frame_info.txt", "w") as text_file:
        text_file.write(str(frame_no))
    
    # For total time taken
    with open("cache/t1.txt", "w") as time_file:
        time_file.write(str(time1))
    
    # Release the capture object and close first session
    video_capture.release()
    yolo.close_session()
    exit(0)

# Takes command line input and passes it    
if __name__ == '__main__':
    run_detector()

# human_detector ends here