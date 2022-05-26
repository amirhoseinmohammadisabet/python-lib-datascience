"""import tensorflow as tf

A = tf.constant([[1, 2],
                 [3, 4]])
print(A)
B = tf.constant([[5, 6],
                 [7, 8]])
print(B)
v = tf.Variable([[1, 2],
                 [2, 4]])
print(v)

AB_concatenated = tf.concat(values=[A, B], axis=1)
print(AB_concatenated)

tensor1 = tf.zeros(shape=[3, 4], dtype=float)
print(tensor1)

tensor2 = tf.ones(shape=[3, 4], dtype=tf.float32)
print(tensor2)

srt = tf.reshape(tensor=tensor1, shape=[4, 3])
print(srt)

tensor3 = tf.constant([[4.6, 4.2],
                       [7.5, 3.6],
                       [2.7, 9.4],
                       [6.7, 8.3]],
                      dtype=tf.float32)

tensor_as_int = tf.cast(tensor3, tf.int32)
print(tensor3)
print(tensor_as_int)

trp = tf.transpose(tensor_as_int)
print(trp)

row, col = trp.shape
print('rows:', row, 'columns:', col)
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense

celsius_q = np.array([-40, 10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
for i, c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

model = Sequential([Dense(units=1, input_shape=[1])])

model.compile(loss='mean_square_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Training is finished")
