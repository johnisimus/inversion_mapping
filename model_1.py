#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 08:45:19 2018

@author: jstockdale
"""
import re
from get_data import get_all_data
from keras import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, TimeDistributed, Bidirectional
from keras.layers.core import Activation

class Model():

    def __init__(self):
        self.X_train = []
        self.Y_train = []
        self.model = Sequential()
        
    def add_training_files(self, path):
        with open(path, 'r') as f:
            training_files = f.readlines()
        pattern = re.compile('mngu0_s1_(\d{4})\n')
        for filename in training_files:
            num = int(pattern.match(filename).group(1))
            self.X_train.append(self.lsfs[num])
            self.Y_train.append(self.emas[num])
    
    def load_lsf_data(self, path):
        self.lsfs = get_all_data(path, 'lsf')
    
    def load_ema_data(self, path):
        self.emas = get_all_data(path, 'ema')
    
    def define_model(self, hidden_size, input_size):
        self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True),input_shape=(None, input_size)))
        self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(12)))
        self.model.add(Activation('sigmoid'))
        print('compiling')
        self.model.compile(loss = 'mean_squared_error', optimizer = 'adam', 
                               metrics = ['accuracy'])
    
    def data_generator(self):
        while True:
            for x,y in zip(self.X_train, self.Y_train):
                yield (x.reshape(1, len(x), len(x[0])), y.reshape(1, len(y),
                                 len(y[0])))

    def train_model(self):
        data_gen = self.data_generator()
        self.model.fit_generator(data_gen, steps_per_epoch=len(self.X_train), epochs=10)
    
