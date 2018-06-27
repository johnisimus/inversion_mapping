#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 08:45:19 2018

@author: jstockdale
"""
import re
import numpy as np
from inv_functions import get_all_data, pad_3D_sequence
from keras import Sequential
from keras.models import model_from_json
from keras.layers.recurrent import LSTM
from keras.layers import Dense, TimeDistributed, Bidirectional
from keras.layers.core import Activation


class Model():

    def __init__(self):
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        
    def add_training_files(self, path):
        with open(path, 'r') as f:
            training_files = f.readlines()
        pattern = re.compile('mngu0_s1_(\d{4})\n')
        for filename in training_files:
            num = int(pattern.match(filename).group(1))
            self.X_train.append(self.lsfs[num])
            self.Y_train.append(self.emas[num])
    
    def add_test_files(self, path):
        with open(path, 'r') as f:
            test_files = f.readlines()
        pattern = re.compile('mngu0_s1_(\d{4})\n')
        for filename in test_files:
            num = int(pattern.match(filename).group(1))
            self.X_test.append(self.lsfs[num])
            self.Y_test.append(self.emas[num])
            
    
    def pad_data(self):
        self.X = pad_3D_sequence(self.X_train)
        self.Y = pad_3D_sequence(self.Y_train)
    
    def load_padded_data(self, x_path, y_path):
        self.X = np.load(x_path)
        self.Y = np.load(y_path)
    
    def load_lsf_data(self, path):
        self.lsfs = get_all_data(path, 'lsf')
    
    def load_ema_data(self, path):
        self.emas = get_all_data(path, 'ema')
    
    def define_model(self, hidden_size, input_size):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True),input_shape=(None, input_size)))
        self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(12)))
        self.model.add(Activation('sigmoid'))
        print('compiling')
        self.model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', 
                               metrics = ['accuracy'])
    
    def data_generator(self, test=False):
        while True:
            if test:
                x_list, y_list = self.X_test, self.Y_test
            else:
                x_list, y_list = self.X_train, self.Y_train
            for x,y in zip(x_list, y_list):
                yield (x.reshape(1, len(x), len(x[0])), y.reshape(1, len(y),
                                 len(y[0])))

    def train_model(self, num_epochs):
        data_gen = self.data_generator()
        self.model.fit_generator(data_gen, steps_per_epoch=42, epochs=num_epochs)
    
    def train_model_no_generator(self, num_epochs):
        self.model.fit(self.X, self.Y, epochs=num_epochs, batch_size=42)
    
    
    def load_model(self, path_json, path_h5):
        json_file = open(path_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights('model.h5')
    



#a tiny change