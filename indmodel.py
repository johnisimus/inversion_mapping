#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 12:12:27 2018

@author: s1773886
"""

from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers.recurrent import LSTM
import numpy as np
import pickle

class IndependentModel():
    
    def __init__(self, use_gpu):
        self.use_cudnn = use_gpu
    
    def load_training_data(self, x_data_loc, y_data_loc):
        with open(x_data_loc, 'rb') as f:
            self.X_data = pickle.load(f)
        with open(y_data_loc, 'rb') as f:
            self.Y_data = pickle.load(f)
    
    def define_model(self, num_hidden):
        input_size = self.X_train.shape[2]
        num_timesteps = self.X_train.shape[1]
        visible_1 = Input(shape=(None, num_timesteps, input_size))
        hidden1 = LSTM(num_hidden, return_sequences=True)(visible_1)
        hidden2 = LSTM(num_hidden, return_sequences=True)(hidden1)
        
        