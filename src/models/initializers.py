from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math


def identity_initializer(scale = 1.0):

    def _initializer(shape, dtype=tf.float32, partition_info=None):

        identity = None
        normal_fill = tf.truncated_normal(shape, stddev=0.01)
        if len(shape) == 1:
            identity = tf.constant(0., dtype=dtype, shape=shape)
        elif len(shape) == 2 and shape[0] == shape[1]:
            identity = tf.constant(scale*np.identity(shape[0], dtype))
        elif len(shape) == 4 and shape[2] == shape[3]:
            array = np.zeros(shape, dtype=float)
            cx, cy = shape[0]/2, shape[1]/2
            for i in range(shape[2]):
                array[cx, cy, i, i] = 1
            identity = tf.constant(scale*array, dtype=dtype)
        else:
            identity = tf.eye(shape[1], batch_shape=[shape[0]]) 

        return tf.maximum(identity, normal_fill)
    return _initializer




def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''

    def _initializer(shape, dtype=tf.float32, partition_info=None):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer