#coding:utf-8

import tensorflow as tf
import numpy as np
from numpy.random import RandomState

# 定义训练数据batch_size
batch_size = 8

# 定义神经网络的参数，两层
w1 = tf.Variable(tf.random_normal([2, 5], stddev=1, seed=1, dtype=tf.float32))
w2 = tf.Variable(tf.random_normal([5, 1], stddev=1, seed=1, dtype=tf.float32))
b1 = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b2 = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

# 输入数据占位符
x = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="x_input")
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y_input")


# 前向计算
a = tf.nn.relu(tf.matmul(x, w1) + b1)
y = tf.nn.relu(tf.matmul(a, w2) + b2)


# 定义损失函数和反向传播
cross_entropy = - tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=cross_entropy)

rdm = RandomState(1)
data_size = 128

X = rdm.rand(data_size, 2)
Y = [[int(x1+x2 < 1)] for(x1, x2) in X]

# 创建会话
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print sess.run(b1)
    print sess.run(b1)

    STEPS = 10000
    for i in xrange(STEPS):
        start = (i * batch_size) % data_size
        end = min(start+batch_size, data_size)

        # 训练
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x: X, y_: Y})
            print ("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    print sess.run(w1)
    print sess.run(w2)