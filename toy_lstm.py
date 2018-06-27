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
        self.model.add(LSTM(hidden_size, return_sequences=True, input_shape=(None, input_size)))
        self.model.add(Activation('sigmoid'))
        print('compiling')
        self.model.compile(loss = 'mean_squared_error', optimizer = 'adam', 
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

    def train_model(self):
        data_gen = self.data_generator()
        self.model.fit_generator(data_gen, steps_per_epoch=100, epochs=1)
    

ema_path = '/home/jstockdale/dis_data/mngu0_s1_ema_norm_1.0.1/'
lsf_path = '/home/jstockdale/dis_data/mngu0_s1_lsf_norm_1.0.1/'
training_path = '/home/jstockdale/dis_data/mngu0_s1_ema_filesets/trainfiles.txt'

#model = Model()
#model.load_ema_data(ema_path)
#model.load_lsf_data(lsf_path)
#model.add_training_files(training_path)
#model.define_model(128, 41)
#model.train_model()

#a tiny change