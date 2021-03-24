from __future__ import print_function

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

import copy
import json
import math
import numpy as np
import os
import pickle
import pprint
import random
import sys
import time
import gc

from collections import defaultdict

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# samples_prediction = [] # np.empty((0,), dtype=(np.float32))
# samples_confidence = [] # np.empty((0,), dtype=(np.float32))
# samples_result     = [] # np.empty((0,), dtype=(np.float32))

inputs = []
labels = []

# Three dimensions:
# - X samples. One sample is one sequence. Each sample consists of:
#   - Y timesteps. One timestep represents is a point in time and includes the features in it. Each timestep consists of:
#     - Z features. Each feature is a datum observation.

num_features  =    3 # presence, prediction (from per-sample model), and confidence
num_timesteps =   64 # sequence/sample length
num_samples   = 1024 # number of sequences

# batch_size = 

def gen_data():
    log('Generating data...')

    for i in range(num_samples):
        result = float(random.getrandbits(1))

        num_populated_timesteps = random.randint(num_timesteps // 2, num_timesteps)

        for j in range(num_timesteps):
            if j < num_populated_timesteps:
                presence = 1
                confidence = random.uniform(0, 1)

                if random.uniform(0, 1) < confidence:
                    # Truth
                    prediction = result
                else:
                    prediction = 1 - result
            else:
                presence = 0
                prediction = 0
                confidence = 0

            # samples_result.append(result)
            # samples_confidence.append(confidence)
            # samples_prediction.append(prediction)
            inputs.append([presence, prediction, confidence])
        labels.append([result])

gen_data()
log("OK")

############################################################################################################################################################

log('Creating model...')

# for i in range(10):
#     print(samples_prediction[i], samples_confidence[i], samples_result[i])

inputs = np.array(inputs).reshape((num_samples, num_timesteps, num_features))
labels = np.array(labels).reshape((num_samples, 1))

model = Sequential()
model.add(LSTM(4, input_shape=(num_timesteps, num_features)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

log("OK")

############################################################################################################################################################

cutoff = num_samples * 3 // 4
loss_threshold = 0.001
n_batch = 1

loss = 1
while loss > loss_threshold:
    model.reset_states()
    history = model.fit(inputs[:cutoff], labels[:cutoff], epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
    loss = history.history['loss'][0]

############################################################################################################################################################

predictions = model.predict(inputs[cutoff:], batch_size=n_batch, verbose=1)
good = 0
for i in range(len(labels[cutoff:])):
    if round(predictions[i][0]) == round(labels[cutoff+i][0]):
        good += 1
    else:
        print('=== Failed - expected:', labels[cutoff+i][0], ', got:', predictions[i][0])
        for j in range(num_timesteps):
            print(inputs[cutoff+i][j])
print("Results:", good, "/", len(predictions))
print("Average validation error:", np.sum(np.abs(predictions.flatten()-labels[cutoff:].flatten())) / len(predictions))
