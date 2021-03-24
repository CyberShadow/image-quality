import tensorflow.keras as keras
from tensorflow.keras.layers import *

import numpy as np

num_samples   = 16 * 1024
do_fit        = True

inputs = np.random.rand(num_samples, 2)
labels = inputs[:, 0] + inputs[:, 1]

l_i = Input(shape=(2,))
l_l = Dense(1, activation='linear')
l_o = l_l(l_i)
model = keras.models.Model(
    inputs=[l_i],
    outputs=[l_o],
)
model.compile(loss='mean_squared_error', optimizer='adam')

if do_fit:
    loss = 1
    while loss > 0.00001:
        history = model.fit(inputs, labels,
                            batch_size=256, verbose=1)
        loss = history.history['loss'][0]
else:
    l_l.set_weights([np.array([[1.0], [1.0]]), np.array([0])])

predictions = model.predict(inputs, batch_size=4096, verbose=1)
good = 0
for i in range(num_samples):
    if abs(predictions[i][0] - labels[i]) < 0.01:
        good += 1
    else:
        print('=== Failed - inputs: ', inputs[i] , ', expected:', labels[i], ', got:', predictions[i][0])
print("Results:", good, "/", len(predictions))
