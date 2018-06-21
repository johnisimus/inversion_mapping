#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 08:45:19 2018

@author: jstockdale
"""

import numpy as np
import re
from get_data import get_all_data
from keras import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation

class Model():

    def __init__(self):
        self.tr_numbers = []
        self.lsfs = {}
        self.emas = {}
        self.model = Sequential()
        
    def add_training_files(self, path):
        with open(path, 'r') as f:
            training_files = f.readlines()
        pattern = re.compile('mngu0_s1_(\d{4})\n')
        for filename in training_files:
            num = int(pattern.match(filename).group(1))
            self.tr_numbers.append(num)
    
    def load_lsf_data(self, path):
        self.lsfs = get_all_data(path, 'lsf')
    
    def load_ema_data(self, path):
        self.emas = get_all_data(path, 'ema')
    
    def define_model(self, output_size, input_size):
        self.model.add(LSTM(output_size, input_shape = (None, input_size), 
                            return_sequences=True, stateful=True, batch_input_shape=(1, None, input_size)))
        self.model.add(Activation('sigmoid'))
    
    def train_model(self):
        for t in self.tr_numbers:
            lsf_temp = self.lsfs[t]
            ema_temp = self.emas[t]
            X = lsf_temp.reshape(1, len(lsf_temp), len(lsf_temp[0]))
            Y = ema_temp.reshape(1, len(ema_temp), len(ema_temp[0]))
            self.model.compile(loss = 'mean_squared_error', optimizer = 'adam', 
                               metrics = ['accuracy'])
            self.model.fit(X, Y, batch_size=1)

ema_path = '/home/jstockdale/dis_data/mngu0_s1_ema_norm_1.0.1/'
lsf_path = '/home/jstockdale/dis_data/mngu0_s1_lsf_norm_1.0.1/'
training_path = '/home/jstockdale/dis_data/mngu0_s1_ema_filesets/trainfiles.txt'

model = Model()
model.add_training_files(training_path)
model.load_ema_data(ema_path)
model.load_lsf_data(lsf_path)
model.define_model(12, 41)
model.train_model()