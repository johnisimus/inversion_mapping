#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 11:36:02 2018

@author: jstockdale
"""

import re
import numpy as np
import os
from estfile3 import estfile
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle


#class Data():
#    
#    def __init__(self, x_data_loc, y_data_loc, maxlen, mode):
#        self.x_path = x_data_loc
#        self.y_path = y_data_loc
#        self.X_list = []
#        self.Y_list = []
#        lsfs = get_all_data(x_data_loc, 'lsf')
#        emas = get_all_data(y_data_loc, 'ema')
#        if mode == 'train':
#            X,Y = list_files(lsfs, emas, 'trainfiles.txt')
#        if mode == 'validation':
#            X,Y = list_files(lsfs, emas, 'validationfiles.txt')
#        del lsfs, emas
#        P, L = pad_and_save(X, truncate=True, maxlen=maxlen)
#        self.X_list.append(P)
#        while len(L) > 0:
#            P, L = pad_and_save(L, truncate=True, maxlen=maxlen)
#            self.X_list.append(P)
#        P, L = pad_and_save(Y, truncate=True, maxlen=maxlen)
#        while len(L) > 0:
#            P, L = pad_and_save(L, truncate=True, maxlen=maxlen)
#            self.Y_list.append(P)
#    
#    def save_data(self, xname, yname):
#        with open(xname, 'wb') as f:
#            pickle.dump(self, f)
#        with open(yname, 'wb') as f:
#            pickle.dump(self, f)
def get_lsf_5(path):
    trackfile = estfile()
    trackfile.load(path)

    lsfs = np.array([[]], dtype='float32').reshape(0,41)
    for t in trackfile.D:
        temp = np.array([t[164:205]], dtype='float32')
        lsfs = np.concatenate((lsfs, temp))
    return lsfs

def get_all_data(path, flag, scale=False, feature_range=(-1,1), return_scaler=False):
    files = os.listdir(path)
    d = {}
    if flag == 'lsf':
        pattern = re.compile(r'mngu0_s1_(\d{4}).lsf')
    
    if flag == 'ema':
        pattern= re.compile(r'mngu0_s1_(\d{4}).ema')
        
    for f in files:
        if f[0] == 'm':
            num = int(pattern.match(f).group(1))
            if flag == 'lsf':
                d[num] = get_lsf_5(os.path.join(path, f))
            if flag == 'ema':
                d[num] = get_ema_data(os.path.join(path, f))

    if scale:
        print('scaling...')
        d, scaler = concat_and_transform(d, feature_range, return_scaler=True)
    if return_scaler:
        return d, scaler
    else:
        return d

def get_ema_data(path):
    trackfile = estfile()
    trackfile.load(path)
    
    emas = np.array([[]], dtype='float32').reshape(0,12)
    for t in trackfile.D:
        temp = np.array([t[0:12]], dtype='float32')
        emas = np.concatenate((emas, temp))
        
    return emas

def pad_3D_sequence(seq):
    maxlen = max([len(t) for t in seq])
    newarray = np.array([[[]]], dtype='float32').reshape(0,maxlen,len(seq[0][0]))
    for n, t in enumerate(seq):
        length = len(t)
        for i in range(length, maxlen):
            t = np.concatenate((t, np.zeros((1,len(t[0])))))
        newarray = np.concatenate((newarray,t.reshape(1,len(t),len(t[0]))))
        print('done with part ',n,' ',i)
    return newarray

def pad_and_save(seq, name=None, truncate=False, maxlen=2000, save=False):
    if truncate:
        P = pad_sequences(seq, dtype='float32', truncating='post', padding='post', maxlen=maxlen)
        leftovers = fix_leftovers(seq, maxlen)
        return (P, leftovers)
    else:
        P = pad_sequences(seq, dtype='float32', padding = 'post')
        if save:
            np.save(name, P)
        else:
            return P


def prepare_and_save(data, name):
    master_array = np.array([[]], dtype='float32').reshape(0, len(data[0][0]))
    for a in data:
        zeroed = np.concatenate((a, np.zeros((1, a.shape[1]), dtype='float32')))
        master_array = np.concatenate((master_array,zeroed))
    np.save(name, master_array)

def list_files(all_x_data, all_y_data, path, save=False, name_x=None, name_y=None):
    X_train = []
    Y_train = []
    with open(path, 'r') as f:
        training_files = f.readlines()
    pattern = re.compile('mngu0_s1_(\d{4})\n')
    for filename in training_files:
        num = int(pattern.match(filename).group(1))
        X_train.append(all_x_data[num])
        Y_train.append(all_y_data[num])
    
    if save:
        with open(name_x, 'wb') as f:
            pickle.dump(X_train, f)
        with open(name_y, 'wb') as f:
            pickle.dump(Y_train, f)
    else:
        return X_train, Y_train

def fix_leftovers(data, maxlen):
    leftovers = [a[maxlen:len(a)] for a in data if len(a) > maxlen]
    return leftovers

#def transform_list(data):
#    scaler = MinMaxScaler(feature_range=(-1,1))
#    scaled_data = []
#    for d in data:
#        scaled_data.append(scaler.fit_transform(d))
#    return scaled_data

def concat_and_transform(data, feature_range, fit=False, return_scaler = False):
    long_array = np.array([[]], dtype='float32').reshape(0, data[1].shape[1])
    for i in range(1, len(data)+1):
        long_array = np.concatenate((long_array, data[i]))
    scaler = MinMaxScaler(feature_range)
    if fit:
        scaler.fit(long_array)
        return scaler
    scaled = scaler.fit_transform(long_array)
    new_dict = {}
    i=0
    j=0
    for k in range(1, len(data)+1):
        j += len(data[k])
        new_dict[k] = scaled[i:j]
        i += len(data[k])
    if return_scaler:
        return new_dict, scaler
    else:
        return new_dict
        
def get_mse(model, means, stds, scaled=True, scaler=None):
    
    X_test = model.X_test
    Y_test = model.Y_test
    with open(means, 'r') as f:
        mean_file = f.read()
    means = np.array([float(x) for x in mean_file.split()][0:12])
    
    with open(stds, 'r') as f:
        std_file = f.read()
    stds = np.array([float(x) for x in std_file.split()][0:12])
        
    mses = []
    for x,y in zip(X_test, Y_test):
        y_hat = model.model.predict(x.reshape(1, x.shape[0], x.shape[1]))[0]
        if scaled:
            y_hat = scaler.inverse_transform(y_hat)
        y_hat = un_normalize(y_hat, means, stds)
        y = un_normalize(y, means, stds)
        mses.append(mean_squared_error(y, y_hat))
    return mses

def get_predictions(model, unscale=False, means=None, stds=None):

    X_test = model.X_test
    if unscale:
        with open(means, 'r') as f:
            mean_file = f.read()
        means = np.array([float(x) for x in mean_file.split()][0:12])
        
        with open(stds, 'r') as f:
            std_file = f.read()
        stds = np.array([float(x) for x in std_file.split()][0:12])
    predictions = []
    for x in X_test:
        y_hat = model.model.predict(x.reshape(1, x.shape[0], x.shape[1]))[0]
        if unscale:
            y_hat = un_normalize(y_hat, means, stds)
        predictions.append(y_hat)
    return predictions

def un_normalize(Y, means, stds):
    Y = np.add(np.multiply(4*stds, Y), means)
    return Y
        
            
            
            
            
            