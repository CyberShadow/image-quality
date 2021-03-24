import tensorflow.keras as keras
from tensorflow.keras.layers import *

import numpy as np

num_samples   = 4 * 1024
do_fit        = False

inputs = np.random.rand(num_samples, 2)
labels = (inputs[:, 0] > inputs[:, 1]) * 1.0

l_i = Input(shape=(2,))
l_l = Dense(1, activation='sigmoid')
l_o = l_l(l_i)
model = keras.models.Model(
    inputs=[l_i],
    outputs=[l_o],
)
model.compile(loss='mean_squared_error', optimizer='adam')

if do_fit:
    model.fit(inputs, labels,
              epochs=10, batch_size=1, verbose=1)
else:
    l_l.set_weights([np.array([[ 1e9], [-1e9]]), np.array([0])])

predictions = model.predict(inputs, batch_size=4096, verbose=1)
for i in range(16):
    print([inputs[i], labels[i], predictions[i]])

good = 0
for i in range(num_samples):
    if round(predictions[i][0]) == round(labels[i]):
        good += 1
    else:
        print('=== Failed - inputs: ', inputs[i] , ', expected:', labels[i], ', got:', predictions[i][0])
print("Results:", good, "/", len(predictions))
