#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:45:11 2018

@author: s1773886
"""

from model_2 import Model

model = Model('x_train.npy', 'y_train.npy', batch_size=128)
model.split_data()
model.define_model(200)
model.train(1000)

model_json = model.model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.model.save_weights('model.h5')