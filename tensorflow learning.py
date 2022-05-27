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
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.datasets import mnist

"""
celsius_q = np.array([-40, 10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
for i, c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

model = Sequential([Dense(units=1, input_shape=[1])])

model.compile(loss='mean_square_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Training is finished")
"""


(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
