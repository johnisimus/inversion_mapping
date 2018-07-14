#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:28:51 2018

@author: s1773886
"""

import re
import numpy as np
from inv_functions import *
from keras import Sequential
from keras.models import model_from_json
from keras.layers.recurrent import LSTM
from keras.layers import Dense, TimeDistributed, Bidirectional, Masking
from keras.layers.core import Activation
from keras.layers import CuDNNLSTM
from keras.callbacks import EarlyStopping
from save_data import Data
import pickle
        

class ModelOne():

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
        

class ModelTwo():
    
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
    
    def load_lsf_data(self, path):
        self.lsfs = get_all_data(path, 'lsf')
    
    def load_ema_data(self, path):
        self.emas = get_all_data(path, 'ema')
    
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
    

    def load_model(self, path_json, path_h5):
        json_file = open(path_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(path_h5)
        self.model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    
    def add_test_files(self, path):
        self.X_test = []
        self.Y_test = []
        with open(path, 'r') as f:
            test_files = f.readlines()
        pattern = re.compile('mngu0_s1_(\d{4})\n')
        for filename in test_files:
            num = int(pattern.match(filename).group(1))
            self.X_test.append(self.lsfs[num])
            self.Y_test.append(self.emas[num])
    
    def test_data_generator(self, test=False):
        while True:
            for x,y in zip(self.X_test, self.Y_test):
                yield (x.reshape(1, len(x), len(x[0])), y.reshape(1, len(y),
                                 len(y[0])))

class ModelThree():
    
    def __init__(self, use_gpu=False):
        self.use_cudnn = use_gpu

    def load_data(self, x_train_loc, y_train_loc, x_val_loc, y_val_loc):
        self.X_train = np.load(x_train_loc)
        self.Y_train = np.load(y_train_loc)
        self.val = (np.load(x_val_loc), np.load(y_val_loc))
    
    def define_model(self, hidden_size, test=False):
        if test:
            input_size = self.X_test[0].shape[1]
        else:
            input_size = self.X_train.shape[2]
        self.model = Sequential()
#        self.model.add(Masking(input_shape=(None, input_size)))
        if self.use_cudnn:
            self.model.add(Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True), input_shape=(None, input_size)))
            self.model.add(Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True)))
        else:
            self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True), input_shape=(None, input_size)))
            self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(12, activation=None)))
        print('compiling')
        self.model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', 
                               metrics = ['mse'])
    
    def load_test_data(self, x_test_loc, y_test_loc):
        with open(x_test_loc, 'rb') as f:
            self.X_test = pickle.load(f)
        with open(y_test_loc, 'rb') as f:
            self.Y_test = pickle.load(f)
    
    def test_generator(self):
        while True:
            for x,y in zip(self.X_test, self.Y_test):
                yield (x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1]))
        
    
    def train(self, batch_size, num_epochs):
        early_stopping = EarlyStopping(monitor='val_loss', patience=7)
        self.model.fit(self.X_train, self.Y_train, batch_size=batch_size, 
                       epochs=num_epochs, validation_data=self.val, callbacks=[early_stopping])
    
    def test(self):
        gen = self.test_generator()
        score = self.model.evaluate_generator(gen, steps=len(self.X_test))
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))

class ModelFour():
    
    def __init__(self, x_train_loc, y_train_loc, x_val_loc, y_val_loc, batch_size):
        self.X_train_path = x_train_loc
        self.Y_train_path = y_train_loc
        self.X_val_path = x_val_loc
        self.Y_val_path = y_val_loc
        self.batch_size = batch_size
        
    def load_and_prep_data(self, maxlen):
        with open(self.X_train_path, 'rb') as f:
            X_train_list = pickle.load(f)
        with open(self.Y_train_path, 'rb') as f:
            Y_train_list = pickle.load(f)
        with open(self.X_val_path, 'rb') as f:
            X_val_list = pickle.load(f)
        with open(self.Y_val_path, 'rb') as f:
            Y_val_list = pickle.load(f)
        
        self.X_train = pad_and_save(X_train_list, truncate=True, maxlen=maxlen)
        del X_train_list
        
        self.Y_train = pad_and_save(Y_train_list, truncate=True, maxlen=maxlen)
        del Y_train_list
        
        self.X_val = pad_and_save(X_val_list, truncate=True, maxlen=maxlen)
        del X_val_list
        
        self.Y_val = pad_and_save(Y_val_list, truncate=True, maxlen=maxlen)
    
    def data_generator(self, flag):
        while True:
            i = 0
            j = self.batch_size
            
            if flag == 'train':
                X_data = self.X_train
                Y_data = self.Y_train
            elif flag == 'val':
                X_data = self.X_val
                Y_data = self.Y_val
            x,y = X_data[0][i:j], Y_data[0][i:j]
            yield (x,y)
            length = len(X_data[0])
            while j < length:
                i += self.batch_size
                j += self.batch_size
                x,y = X_data[0][i:j], Y_data[0][i:j]
                yield (x,y)
            
            for x,y in zip(X_data[1], Y_data[1]):
                yield (x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1]))
    
#    def val_data_generator(self):
#        while True:
#            for x,y in zip(self.X_val, self.Y_val):
#                yield (x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1]))
    
    def define_model(self, hidden_size):
        input_size = self.X_train[0].shape[2]
        self.model = Sequential()
        self.model.add(Masking(input_shape=(None, input_size)))
        self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(12)))
        self.model.add(Activation('sigmoid'))
        print('compiling')
        self.model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', 
                               metrics = ['accuracy'])
    
    def train(self, num_epochs):
        train_gen = self.data_generator('train')
        val_gen = self.data_generator('val')
        steps_train = int(len(self.X_train[0]) / self.batch_size) + len(self.X_train[1])
        steps_val = int(len(self.X_val[0]) / self.batch_size) + len(self.X_val[1])
        self.model.fit_generator(train_gen, steps_per_epoch=steps_train,
                                 validation_data = val_gen, validation_steps = steps_val, 
                                 epochs=num_epochs)

class ModelFive():
    
    def __init__(self, train_loc, val_loc, batch_size):
        self.train_loc = train_loc
        self.val_loc = val_loc
        self.batch_size = batch_size

    def load_and_prep_data(self):
        with open(self.train_loc, 'rb') as f:
            self.t_data = pickle.load(f)
        with open(self.val_loc, 'rb') as f:
            self.v_data = pickle.load(f)
        global train_step, val_step
        train_step = 0
        val_step = 0
        for t in self.t_data.X_list:
            train_step += int(len(t) / self.batch_size) + 1
        for t in self.v_data.X_list:
            val_step += int(len(t) / self.batch_size) + 1
        
    def data_generator(self, mode):
        while True:
            if mode == 'train':
                data = self.t_data
            elif mode == 'validation':
                data = self.v_data
            for X, Y in zip(data.X_list, data.Y_list):
                i = 0
                j = self.batch_size
                x, y = X[i:j], Y[i:j]
                yield (x,y)
                length = len(X)
                while j < length:
                    i += self.batch_size
                    j += self.batch_size
                    x,y = X[i:j], Y[i:j]
                    yield (x,y)

    def define_model(self, hidden_size):
        input_size = self.t_data.X_list[0].shape[2]
        self.model = Sequential()
        self.model.add(Masking(input_shape=(None, input_size)))
        self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(12, activation='sigmoid')))
        print('compiling')
        self.model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', 
                               metrics = ['mse'])
    
    def train(self, num_epochs):
        train_gen = self.data_generator('train')
        val_gen = self.data_generator('validation')
        self.model.fit_generator(train_gen, steps_per_epoch=train_step, 
                                 validation_data=val_gen, validation_steps=val_step,
                                 epochs=num_epochs)

class ModelSix():
    def __init__(self, x_train_loc, y_train_loc, x_val_loc, y_val_loc, use_cudnn=False):
        with open(x_train_loc, 'rb') as f:
            self.X_train = pickle.load(f)
        with open(y_train_loc, 'rb') as f:
            self.Y_train = pickle.load(f)
        with open(x_val_loc, 'rb') as f:
            self.X_val = pickle.load(f)
        with open(y_val_loc, 'rb') as f:
            self.Y_val = pickle.load(f)
            
        self.use_cudnn = use_cudnn

    def define_model(self, hidden_size):
        input_size = self.X_train[0].shape[1]
        self.model = Sequential()
#        self.model.add(Masking(input_shape=(None, input_size)))
        if self.use_cudnn:
            self.model.add(Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True), input_shape=(None, input_size)))
            self.model.add(Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True)))
        else:
            self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True), input_shape=(None, input_size)))
            self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(12, activation=None)))
        print('compiling')
        self.model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', 
                               metrics = ['mse'])
    
    def load_model(self, loc_h5):
        self.define_model(200)
        self.model.load_weights(loc_h5)
        #self.model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['rmsprop'])
        print(self.model.summary())
    
    def load_test_data(self, x_test_loc, y_test_loc):
        with open(x_test_loc, 'rb') as f:
            self.X_test = pickle.load(f)
        with open(y_test_loc, 'rb') as f:
            self.Y_test = pickle.load(f)
    
    def data_generator(self, mode):
        while True:
            if mode == 'train':
                for x, y in zip(self.X_train, self.Y_train):
                    yield (x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1]))
            if mode == 'val':
                for x, y in zip(self.X_val, self.Y_val):
                    yield (x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1]))
            if mode == 'test':
                for x, y in zip(self.X_test, self.Y_test):
                    yield (x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1]))
            
    
    def train(self, num_epochs):
        train_gen = self.data_generator('train')
        val_gen = self.data_generator('val')
        self.model.fit_generator(train_gen, steps_per_epoch=len(self.X_train), 
                                 epochs=num_epochs, validation_data=val_gen,
                                 validation_steps=len(self.X_val))
    
    def test(self):
        test_gen = self.data_generator('test')
        score = self.model.evaluate_generator(test_gen, steps=len(self.X_test))
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))

class FBB():
    
    def __init__(self, use_gpu=False):
        self.use_cudnn = use_gpu

    def load_data(self, x_train_loc, y_train_loc, x_val_loc, y_val_loc):
        self.X_train = np.load(x_train_loc)
        self.Y_train = np.load(y_train_loc)
        self.val = (np.load(x_val_loc), np.load(y_val_loc))
    
    def define_model(self, hidden_size, test=False):
        if test:
            input_size = self.X_test[0].shape[1]
        else:
            input_size = self.X_train.shape[2]
        self.model = Sequential()
        self.model.add(TimeDistributed(Dense(hidden_size, activation='sigmoid'), input_shape=(None, input_size)))
#        self.model.add(Masking(input_shape=(None, input_size)))
        if self.use_cudnn:
            self.model.add(Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True)))
            self.model.add(Bidirectional(CuDNNLSTM(6, return_sequences=True)))
        else:
            self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True), input_shape=(None, input_size)))
            self.model.add(Bidirectional(LSTM(6, return_sequences=True)))
        
        print('compiling')
        self.model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', 
                               metrics = ['mse'])
    
    def load_test_data(self, x_test_loc, y_test_loc):
        with open(x_test_loc, 'rb') as f:
            self.X_test = pickle.load(f)
        with open(y_test_loc, 'rb') as f:
            self.Y_test = pickle.load(f)
    
    def test_generator(self):
        while True:
            for x,y in zip(self.X_test, self.Y_test):
                yield (x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1]))
        
    
    def train(self, batch_size, num_epochs):
        early_stopping = EarlyStopping(monitor='val_loss', patience=7)
        self.model.fit(self.X_train, self.Y_train, batch_size=batch_size, 
                       epochs=num_epochs, validation_data=self.val, callbacks=[early_stopping])
    
    def test(self):
        gen = self.test_generator()
        score = self.model.evaluate_generator(gen, steps=len(self.X_test))
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))

class FBBF():
    
    def __init__(self, use_gpu=False):
        self.use_cudnn = use_gpu

    def load_data(self, x_train_loc, y_train_loc, x_val_loc, y_val_loc):
        self.X_train = np.load(x_train_loc)
        self.Y_train = np.load(y_train_loc)
        self.val = (np.load(x_val_loc), np.load(y_val_loc))
    
    
    def define_model(self, hidden_size, test=False):
        if test:
            input_size = self.X_test[0].shape[1]
        else:
            input_size = self.X_train.shape[2]
        self.model = Sequential()
        self.model.add(TimeDistributed(Dense(hidden_size, activation='sigmoid'), input_shape=(None, input_size)))
#        self.model.add(Masking(input_shape=(None, input_size)))
        if self.use_cudnn:
            self.model.add(Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True)))
            self.model.add(Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True)))
        else:
            self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True), input_shape=(None, input_size)))
            self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
        
        self.model.add(TimeDistributed(Dense(12, activation=None)))
        print('compiling')
        self.model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', 
                               metrics = ['mse'])
    
    def load_test_data(self, x_test_loc, y_test_loc):
        with open(x_test_loc, 'rb') as f:
            self.X_test = pickle.load(f)
        with open(y_test_loc, 'rb') as f:
            self.Y_test = pickle.load(f)
    
    def test_generator(self):
        while True:
            for x,y in zip(self.X_test, self.Y_test):
                yield (x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1]))
        
    
    def train(self, batch_size, num_epochs):
        early_stopping = EarlyStopping(monitor='val_loss', patience=7)
        self.model.fit(self.X_train, self.Y_train, batch_size=batch_size, 
                       epochs=num_epochs, validation_data=self.val, callbacks=[early_stopping])
    
    def test(self):
        gen = self.test_generator()
        score = self.model.evaluate_generator(gen, steps=len(self.X_test))
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))

class FBBF_1():
    
    def __init__(self, use_gpu=False):
        self.use_cudnn = use_gpu    
    
    def define_model(self, hidden_size, test=False):
        if test:
            input_size = self.X_test[0].shape[1]
        else:
            input_size = self.X_train[0].shape[1]
        self.model = Sequential()
        self.model.add(TimeDistributed(Dense(hidden_size, activation='sigmoid'), input_shape=(None, input_size)))
#        self.model.add(Masking(input_shape=(None, input_size)))
        if self.use_cudnn:
            self.model.add(Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True)))
            self.model.add(Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True)))
        else:
            self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True), input_shape=(None, input_size)))
            self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
        
        self.model.add(TimeDistributed(Dense(12, activation=None)))
        print('compiling')
        self.model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', 
                               metrics = ['mse'])
    
    def load_data(self, x_train_loc, y_train_loc, x_val_loc, y_val_loc):
        with open(x_train_loc, 'rb') as f:
            self.X_train = pickle.load(f)
        with open(y_train_loc, 'rb') as f:
            self.Y_train = pickle.load(f)
        with open(x_val_loc, 'rb') as f:
            self.X_val = pickle.load(f)
        with open(y_val_loc, 'rb') as f:
            self.Y_val = pickle.load(f)
            
    def load_test_data(self, x_test_loc, y_test_loc):
        with open(x_test_loc, 'rb') as f:
            self.X_test = pickle.load(f)
        with open(y_test_loc, 'rb') as f:
            self.Y_test = pickle.load(f)
    
    def test_generator(self):
        while True:
            for x,y in zip(self.X_test, self.Y_test):
                yield (x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1]))

    def data_generator(self, mode):
        while True:
            if mode == 'train':
                for x, y in zip(self.X_train, self.Y_train):
                    yield (x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1]))
            if mode == 'val':
                for x, y in zip(self.X_val, self.Y_val):
                    yield (x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1]))
            if mode == 'test':
                for x, y in zip(self.X_test, self.Y_test):
                    yield (x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1]))
    
    def train(self, num_epochs):
        early_stopping = EarlyStopping(monitor='val_loss', patience=7)
        train_gen = self.data_generator('train')
        val_gen = self.data_generator('val')
        self.model.fit_generator(train_gen, steps_per_epoch=len(self.X_train), 
                                 epochs=num_epochs, validation_data=val_gen,
                                 validation_steps=len(self.X_val), callbacks=[early_stopping])
    
    def test(self):
        gen = self.test_generator()
        score = self.model.evaluate_generator(gen, steps=len(self.X_test))
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))
        

class GenModel():

    def __init__(self, use_gpu=False):
        self.use_cudnn = use_gpu

    def load_data(self, x_train_loc, y_train_loc, x_val_loc, y_val_loc):
        self.X_train = np.load(x_train_loc)
        self.Y_train = np.load(y_train_loc)
        self.val = (np.load(x_val_loc), np.load(y_val_loc))
    
    def load_test_data(self, x_test_loc, y_test_loc):
        with open(x_test_loc, 'rb') as f:
            self.X_test = pickle.load(f)
        with open(y_test_loc, 'rb') as f:
            self.Y_test = pickle.load(f)
    
    def test_generator(self):
        while True:
            for x,y in zip(self.X_test, self.Y_test):
                yield (x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1]))
        
    
    def train(self, batch_size, num_epochs):
        early_stopping = EarlyStopping(monitor='val_loss', patience=7)
        self.model.fit(self.X_train, self.Y_train, batch_size=batch_size, 
                       epochs=num_epochs, validation_data=self.val, callbacks=[early_stopping])
    
    def test(self):
        gen = self.test_generator()
        score = self.model.evaluate_generator(gen, steps=len(self.X_test))
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))

class FBF(GenModel):
    
    def __init__(self, use_gpu=False):
        self.use_cudnn = super().__init__(use_gpu)
    
    
    def define_model(self, hidden_size, test=False):
        if test:
            input_size = self.X_test[0].shape[1]
        else:
            input_size = self.X_train[0].shape[1]
        self.model = Sequential()
        self.model.add(TimeDistributed(Dense(hidden_size, activation='sigmoid'), input_shape=(None, input_size)))
#        self.model.add(Masking(input_shape=(None, input_size)))
        if self.use_cudnn:
            self.model.add(Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True)))
        else:
            self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True), input_shape=(None, input_size)))
        
        self.model.add(TimeDistributed(Dense(12, activation=None)))
        print('compiling')
        self.model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', 
                               metrics = ['mse'])
