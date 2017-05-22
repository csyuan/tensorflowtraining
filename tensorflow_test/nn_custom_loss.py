#coding:utf-8

import numpy as np
from numpy.random import RandomState
import tensorflow as tf


# v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
# v2 = tf.constant([4.0, 3.0, 2.0, 1.0])

sess = tf.InteractiveSession()

# print tf.greater(v1, v2).eval()
# # tf.select替换成tf.where
# print tf.where(tf.greater(v1, v2), v1, v2).eval()

batch_size = 8
x = tf.placeholder(tf.float32, shape=(None, 2), name='x_input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')

w1 = tf.Variable(tf.random_normal([2, 1], stddev=1.0, seed=1, dtype=tf.float32))
y = tf.matmul(x, w1)

loss_less = 10
loss_more = 1

loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

rdm = RandomState(1)
data_size = 128
X = rdm.rand(data_size, 2)
Y = [[x1 + x2 + rdm.rand() /10.0 - 0.05] for(x1, x2) in X]

tf.global_variables_initializer().run()
STEPS = 5000
for i in range(STEPS):
    start = (i * batch_size) % data_size
    end = min(start+batch_size, data_size)
    sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

print sess.run(w1)
print sess.run(loss, feed_dict={x: X, y_: Y})

sess.close()

