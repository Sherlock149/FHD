# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:07:50 2021

@author: abhishek
"""
"""
This is the core module that loads the MoveNet model and runs inference.
It puts the cropped image into the model and returns all detected keypoints.
Whole body keypoints along with their confidence scores are returned.

"""
# Importing the required modules
import cv2
import tensorflow as tf

# Defining the MoveNet class
class MoveNet(object):
    
    # Constructor to load the model and allocate tensors
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(model_path="weights/movenet_lightning.tflite")
        self.input_size = 192
        self.interpreter.allocate_tensors()
       
    # Function to carry out prediction
    def predictor(self,bbox):
        
        # Preprocessing
        rgb = cv2.cvtColor(bbox, cv2.COLOR_BGR2RGB)
        img_tensor = tf.convert_to_tensor(rgb, dtype=(tf.uint8))
        frame = tf.expand_dims(img_tensor, axis=0)
        frame = tf.image.resize_with_pad(frame, self.input_size, self.input_size)
        frame = tf.cast(frame, dtype=(tf.float32))
        
        # Getting tensor details and setting the interpreter accordingly
        inputs = self.interpreter.get_input_details()
        outs = self.interpreter.get_output_details()
        self.interpreter.set_tensor(inputs[0]['index'], frame.numpy())
        
        # Predicting the keypoints using the model and collecting the output
        self.interpreter.invoke()
        keypoints = self.interpreter.get_tensor(outs[0]['index'])
        
        # Returning the detected keypoints
        return keypoints
    
# MoveNet_core ends here