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
            length = len(self.x_list)
            yield (x,y)
            while j < len(self.x_list):
                i += self.batch_size
                j += self.batch_size
                if j < length:
                    x,y = np.asarray(self.x_list[i:j]), np.asarray(self.y_list[i:j])
                else:
                    x,y = np.asarray(self.x_list[i:length-1]), np.asarray(self.y_list[i:length-1])
                    yield(x,y)
                    x,y = np.array([self.x_list[-1]]), np.array([self.y_list[-1]])
                yield (x,y)
    
    def define_model(self, hidden_size):
        input_size = self.x_list[0].shape[1]
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True),input_shape=(None, input_size)))
        self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(12)))
        self.model.add(Activation('sigmoid'))
        print('compiling')
        self.model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', 
                               metrics = ['accuracy'])
    
    def train(self, num_epochs):
        gen = self.data_generator()
        steps = int(len(self.x_list) / self.batch_size)
        self.model.fit_generator(gen, steps_per_epoch=steps, epochs=num_epochs)
        
                
                
                

            