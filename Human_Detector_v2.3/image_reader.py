# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 16:10:50 2021

@author: abhishek
"""
"""
This module is for displaying bbox anootations from ndjson file on image

"""
# Importing the required modules
import ndjson
import cv2

file_path = 'test_dataset/test_image.jpg'

with open('output/test_image-coordinates.ndjson') as json_file:
    data = ndjson.load(json_file)

image = cv2.imread(file_path)

for bbox in data[0]['people']:
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    
cv2.imshow('preview', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# image_reader ends here