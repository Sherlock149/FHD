# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:22:22 2021

@author: abhishek
"""
"""
This is the core yolo module that is called upon whenever human_detector runs
It contains major functions to carry out the detection using a keras model and
return bounding box coordinates, confidence scores and class names

Reference: https://github.com/Ma-Dan/keras-yolo4/blob/master/yolo4.py

"""
# Importing the required modules
import tensorflow as tf
from tensorflow import keras
import tensorflow.compat.v1.keras.backend as K
import numpy as np
import os
import ntpath
from tensorflow.keras.models import load_model
from utils import Mish,yolo_eval,letterbox_image

# Else it raises error
tf.compat.v1.disable_eager_execution()

# main class
class Detector(object):
    # Constructor to initialise various parameters and call generate() to
    # load the model and generate the tensors
    def __init__(self,gpu_number,cs):
        self.model_path = 'weights/yolo4_weight.h5'
        self.anchors_path = 'weights/yolo_anchors.txt'
        self.classes_path = 'weights/coco_classes.txt'
        # Multi GPU
        self.gpu_num = int(gpu_number)
        # cs threshold
        self.score = cs
        # iou threshold for nms
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.boxes, self.scores, self.classes = self.generate()
    
    # Getting the coco class names
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    # Loading the anchors
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors
    
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        
        # Loading the model with the mish activation function
        # compile = False => Not trainable
        self.yolo_model = load_model(model_path, custom_objects={'Mish': Mish}, compile=False)
        print("Model: "+ ntpath.basename(model_path) + " loaded...")

        # Generate colors for drawing bounding boxes.
        self.colors = (0,255,0)

        # Generate output tensor targets for filtered bounding boxes.
        # Threshold and other parameters are defined here
        self.input_image_shape = K.placeholder(shape=(2, ))
        
        # If multiple GPUs are present, prepare parallel model
        if self.gpu_num >= 2:
            multiGPU = tf.distribute.MirroredStrategy()
            print("Found: {} GPUs".format(multiGPU.num_replicas_in_sync))
            with multiGPU.scope():
                self.yolo_model = self.yolo_model
                print("Parallel Model Ready!")
        
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_from_frame(self, image):
        # Data preprocessing including feature scaling and converting it ot 4D array
        new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)
        
        # Put the image into the model and get the output coordinates
        # Session is run as a validation phase to prevent training
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        
        # Extracting the bbox coordinates, class name and confidence scores from
        # output and returning them after processing    
        bbox = []
        classes = []
        confidence_score = []
        for i, clas in reversed(list(enumerate(out_classes))):
            class_name = self.class_names[clas]
            # For only detecting humans
            if class_name == 'person':
                box = out_boxes[i]
                cs = out_scores[i]
                x = int(box[1])
                y = int(box[0])
                w = int(box[3] - box[1])
                h = int(box[2] - box[0])
                if x < 0:
                    w = w + x
                    x = 0
                if y < 0:
                    h = h + y
                    y = 0
                bbox.append([x, y, w, h])
                confidence_score.append(cs)
                classes.append(class_name)

        return bbox, confidence_score, classes

    def close_session(self):
        self.sess.close()

# yolov4_core ends here