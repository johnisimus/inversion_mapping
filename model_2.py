#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 09:54:15 2018

@author: jstockdale
"""

from keras import Sequential
from keras.models import model_from_json
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Bidirectional, TimeDistributed
from keras.layers.core import Activation
import numpy as np

class Model():
    
    def __init__(self, x_data_loc, y_data_loc, timestep_size = 100, batch_size = 32):
        self.X_path = x_data_loc
        self.Y_path = y_data_loc
        self.timestep_size = timestep_size
        self.batch_size = batch_size
    
    def split_data(self):
        X = np.load(self.X_path)
        Y = np.load(self.Y_path)
        self.x_list = []
        self.y_list = []
        i = 0
        j = self.timestep_size
        self.x_list.append(X[i:j])
        self.y_list.append(Y[i:j])
        while j < len(X):
            i += self.timestep_size
            j += self.timestep_size
            self.x_list.append(X[i:j])
            self.y_list.append(Y[i:j])
        del X,Y
    
    def data_generator(self):
        while True:
            i = 0
            j = self.batch_size
            x,y = np.asarray(self.x_list[i:j]), np.asarray(self.y_list[i:j])
            yield (x,y)
            while j < len(self.x_list):
                i += self.batch_size
                j += self.batch_size
                x,y = np.asarray(self.x_list[i:j]), np.asarray(self.y_list[i:j])
                yield (x,y)
                
                
                

            