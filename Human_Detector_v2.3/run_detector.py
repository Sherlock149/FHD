# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:20:34 2021

@author: abhishek
"""
"""
This is the main script to be run by the user. It automatically conducts human
detection and scaling based on the CLI input arguments.

"""
# Importing the required modules
import ntpath
import os
import click
import time
import ndjson
from collections import OrderedDict

# CLI
@click.command()
@click.option('--file-path', help='Input video filepath')
@click.option('--gpu-num', default=1, help='Number of GPUs. Default: 1')
@click.option('--include-score', is_flag=True, help='Include confidence score in ouput. Default: False')
@click.option('--threshold', default=0.4, help='Confidence Score threshold. Default: 0.4')
@click.option('--position-threshold', default=0.11, help='Position Score threshold. Default: 0.11')
@click.option('--skip-scaling', is_flag=True, help='Skip scaling. Default: False')

# main function
def run_main(file_path,gpu_num,include_score,threshold,position_threshold,skip_scaling):
    
    # writer function needed in case smoothing is enabled
    def ndjson_creator(data,total_frames,outpt_file):
        with open('output/'+outpt_file + '-coordinates_with_keypoints.ndjson', 'w') as outfile:
            writer = ndjson.writer(outfile, ensure_ascii=False)
            
            # Creating one row for each frame
            for i in range(total_frames):
                dt = OrderedDict()
                dt["frame_no"] = i
                dt["people"] = []
                
                # Traversing through all the json data
                for box in data[i]['people']:
                    try:
                        dt["people"].append([box[0],box[1],box[2],box[3],box[4],box[5],box[6],box[7]])
                        
                    except IndexError:
                        dt["people"].append([box[0],box[1],box[2],box[3],box[4],box[5]])
                    
                # Writing the row into the ndjson file
                writer.writerow(dt)
    
    if skip_scaling == False:
        flag = False
        file_name1 = ntpath.basename(file_path)
        file_name = os.path.splitext(file_name1)[0]
        img_check = os.path.splitext(file_name1)[1]
        
        # If input is image
        if img_check == ".jpg" or img_check == ".png":
            # Run Human Detection for image without scaling
            if include_score:
                os.system('python human_detector_v2.py --file-path '+file_path+' --gpu-num '+str(gpu_num)+' --include-score'+' --threshold '+str(threshold))
            else:
                os.system('python human_detector_v2.py --file-path '+file_path+' --gpu-num '+str(gpu_num)+' --threshold '+str(threshold))
            exit(0)
    
        # If input is video   
        try:
            # Run Human Detection
            if include_score:
                os.system('python human_detector_v2.py --file-path '+file_path+' --gpu-num '+str(gpu_num)+' --include-score'+' --threshold '+str(threshold))
            else:
                os.system('python human_detector_v2.py --file-path '+file_path+' --gpu-num '+str(gpu_num)+' --threshold '+str(threshold))
            
            # Scaling and foot keypoints generation
            print("Starting Scaling...")
            
            # Reading frames to process from cache
            with open("cache/frame_info.txt", "r") as text_file:
                frame_no = text_file.read()
                
            # Scaling has started so if aborted now, program will quit
            flag = True
            if include_score:
                os.system('python foot_detector.py --file-path '+file_path+' --total-frames '+str(frame_no)+' --include-score'+' --threshold '+str(position_threshold))
            else:
                os.system('python foot_detector.py --file-path '+file_path+' --total-frames '+str(frame_no)+' --threshold '+str(position_threshold))
       
        except KeyboardInterrupt:
            # If human detection is stopped midway, proceed to scaling
            if flag == False:
                print("Starting Scaling...")
                with open("cache/frame_info.txt", "r") as text_file:
                    frame_no = text_file.read()
                flag = True
                if include_score:
                    os.system('python foot_detector.py --file-path '+file_path+' --total-frames '+str(frame_no)+' --include-score'+' --threshold '+str(position_threshold))
                else:
                    os.system('python foot_detector.py --file-path '+file_path+' --total-frames '+str(frame_no)+' --threshold '+str(position_threshold))

            
            else:
                print("Aborted!")
        
        # For total time taken
        print(" ")
        with open("cache/t1.txt", "r") as time1:
            a = time1.read()
        with open("cache/t2.txt", "r") as time2:
            b = time2.read()
        seconds = int(a)+int(b)
            
        # Clearing the cache
        os.remove('cache/'+file_name+'-coordinates.ndjson')
        os.remove('cache/frame_info.txt')
        os.remove('cache/t1.txt')
        os.remove('cache/t2.txt')
        os.rmdir("cache/")
        
        time_track = time.time()
        # Printing the total time
        print(" ")
        seconds += round(time.time()-time_track)
        hr = int(int(seconds)/3600)
        seconds = int(seconds - hr*3600)
        mins = int(int(seconds)/60)
        seconds = int(seconds - mins*60)
        print("Total processing time: "+"%02d" % (hr)+":"+"%02d" % (mins)+":"+"%02d" % (seconds))
    
    else:
        
        flag = False
        file_name1 = ntpath.basename(file_path)
        file_name = os.path.splitext(file_name1)[0]
        img_check = os.path.splitext(file_name1)[1]
    
        # If input is video   
        try:
            # Run Human Detection
            if include_score:
                os.system('python human_detector_v2.py --file-path '+file_path+' --gpu-num '+str(gpu_num)+' --include-score'+' --threshold '+str(threshold)+' --skip-scaling')
            else:
                os.system('python human_detector_v2.py --file-path '+file_path+' --gpu-num '+str(gpu_num)+' --threshold '+str(threshold)+' --skip-scaling')
            
            # Foot keypoints generation
            print("Starting Foot Position...")
            
            # Reading frames to process from cache
            with open("cache/frame_info.txt", "r") as text_file:
                frame_no = text_file.read()
                
            # Position detection has started so if aborted now, program will quit
            flag = True
            if include_score:
                os.system('python foot_detector_ns.py --file-path '+file_path+' --total-frames '+str(frame_no)+' --include-score'+' --threshold '+str(position_threshold))
            else:
                os.system('python foot_detector_ns.py --file-path '+file_path+' --total-frames '+str(frame_no)+' --threshold '+str(position_threshold))
       
        except KeyboardInterrupt:
            # If human detection is stopped midway, proceed without scaling
            if flag == False:
                print("Starting Foot Position...")
                with open("cache/frame_info.txt", "r") as text_file:
                    frame_no = text_file.read()
                flag = True
                if include_score:
                    os.system('python foot_detector_ns.py --file-path '+file_path+' --total-frames '+str(frame_no)+' --include-score'+' --threshold '+str(position_threshold))
                else:
                    os.system('python foot_detector_ns.py --file-path '+file_path+' --total-frames '+str(frame_no)+' --threshold '+str(position_threshold))

            
            else:
                print("Aborted!")
        
        # For total time taken
        print(" ")
        with open("cache/t1.txt", "r") as time1:
            a = time1.read()
        with open("cache/t2.txt", "r") as time2:
            b = time2.read()
        seconds = int(a)+int(b)
            
        # Clearing the cache
        os.remove('cache/'+file_name+'-coordinates.ndjson')
        os.remove('cache/frame_info.txt')
        os.remove('cache/t1.txt')
        os.remove('cache/t2.txt')
        os.rmdir("cache/")
        
        time_track = time.time()
        # Printing the total time
        print(" ")
        seconds += round(time.time()-time_track)
        hr = int(int(seconds)/3600)
        seconds = int(seconds - hr*3600)
        mins = int(int(seconds)/60)
        seconds = int(seconds - mins*60)
        print("Total processing time: "+"%02d" % (hr)+":"+"%02d" % (mins)+":"+"%02d" % (seconds))

        
# Takes command line input and passes it    
if __name__ == '__main__':
    run_main(standalone_mode = False)

# run_detector ends here