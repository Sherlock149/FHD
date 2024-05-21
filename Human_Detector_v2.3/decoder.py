# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 07:55:08 2021

@author: abhishek
"""
"""
This module contains function to convert bbox coordinates from (top-left,width,height)
format to opencv format (top-left,bottom_right)

"""
import numpy as np


class Decoder(object):

    # Constructor
    def __init__(self, tlwh, confidence, clas):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.clas = clas

    # Convert to (top-left,bottom_right) format
    def tlbr(self):

        bbox = self.tlwh.copy()
        bbox[2:] += bbox[:2]
        return bbox

# decoder ends here
