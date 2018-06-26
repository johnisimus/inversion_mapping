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


def get_lsf_7(path):
    trackfile = estfile()
    trackfile.load(path)

    lsfs = np.array([[]], dtype='float32').reshape(0,41)
    for t in trackfile.D:
        temp = np.array([t[286:327]], dtype='float32')
        lsfs = np.concatenate((lsfs, temp))
    return lsfs

def get_all_data(path, flag):
    files = os.listdir(path)
    pattern= re.compile(r'mngu0_s1_(\d{4}).lsf')
    d = {}
    if flag == 'lsf':
        pattern = re.compile(r'mngu0_s1_(\d{4}).lsf')
    
    if flag == 'ema':
        pattern= re.compile(r'mngu0_s1_(\d{4}).ema')
        
    for f in files:
        if f[0] == 'm':
            num = int(pattern.match(f).group(1))
            if flag == 'lsf':
                d[num] = get_lsf_7(os.path.join(path, f))
            if flag == 'ema':
                d[num] = get_ema_data(os.path.join(path, f))
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
 

