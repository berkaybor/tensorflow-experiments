"""
A simple test which measures time taken while taking the dot product of
a random matrix with itself.
"""

import numpy as np
import tensorflow as tf
from datetime import datetime

device_name = '/gpu:0' # /gpu:0 or /cpu:0
shape = (10000, 10000)

with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)

start_time = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True,
                allow_soft_placement = True)) as session:
    result = session.run(sum_operation)
    print(result)

print("Time taken:", datetime.now() - startTime)
print("Shape:", shape, "Device:", device_name)
