import tensorflow.keras as keras
from tensorflow.keras.layers import *

import numpy as np

num_samples    = 64 * 1024
do_fit_add     = True
do_fit_compare = False

inputs = np.random.rand(num_samples, 2, 2)
labels = ((inputs[:, 0, 0] + inputs[:, 0, 1]) >
          (inputs[:, 1, 0] + inputs[:, 1, 1])) * 1.0

k_inputs = [inputs[:, i] for i in range(2)]

L_add = Dense(1, activation='linear')

l_inp1 = Input(shape=(2,))
l_inp2 = Input(shape=(2,))

l_add1 = L_add(l_inp1)
l_add2 = L_add(l_inp2)

L_con = Concatenate()
l_con = L_con([l_add1, l_add2])

L_cmp = Dense(1, activation='sigmoid')
l_cmp = L_cmp(l_con)

l_out = l_cmp

model = keras.models.Model(
    inputs=[l_inp1, l_inp2],
    outputs=[l_out],
)

if not do_fit_add:
    L_add.set_weights([np.array([[1.0], [1.0]]), np.array([0])])
    L_add.trainable = False
if not do_fit_compare:
    r = 1e3
    L_cmp.set_weights([np.array([[r], [-r]]), np.array([0])])
    L_cmp.trainable = False

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

if do_fit_add or do_fit_compare:
    loss = 1
    while loss > 0.001:
        history = model.fit(k_inputs, labels,
                            batch_size=64, verbose=1)
        loss = history.history['loss'][0]

predictions = model.predict(k_inputs, batch_size=4096, verbose=1)
good = 0
for i in range(num_samples):
    if round(predictions[i][0]) == round(labels[i]):
        good += 1
    else:
        print('=== Failed - inputs: ', inputs[i] , ', expected:', labels[i], ', got:', predictions[i][0])
print("Results:", good, "/", len(predictions))

print("Adder weights:")
print(L_add.get_weights())
print("Comparator weights: ")
print(L_cmp.get_weights())
