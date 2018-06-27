#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:14:59 2018

@author: jstockdale
"""

from model_1 import Model

ema_path = 'mngu0_s1_ema_norm_1.0.1/'
lsf_path = 'mngu0_s1_lsf_norm_1.0.1'
training_path = 'trainfiles.txt'

model = Model()
model.load_ema_data(ema_path)
model.load_lsf_data(lsf_path)
model.add_training_files(training_path)
model.define_model(200, 41)
model.train_model(num_epochs=50)

model_json = model.model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.model.save_weights('model.h5')

