import numpy as np
import tensorflow as tf

SIZE = 6
CLASS = 10
label = tf.placeholder(tf.int32, (10,))
sess = tf.Session()
print('label1:', sess.run(label, feed_dict={label: [0, 1, 2, 3, 4, 5, 6, 7,8,9]}))
b = tf.one_hot(label, CLASS, 1, 0)
print('after one_hot:\n', sess.run(b, feed_dict={label: [0, 1, 2, 3, 4, 5, 6, 7,8,9]}))

