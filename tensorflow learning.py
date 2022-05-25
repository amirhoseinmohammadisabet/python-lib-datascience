import warnings
import math
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

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
