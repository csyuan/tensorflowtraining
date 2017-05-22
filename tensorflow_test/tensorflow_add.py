#coding:utf-8

import tensorflow as tf

with tf.name_scope("input1"):
    input1 = tf.constant([1.0,2.,3.], name='input1')
with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_uniform([3]), name="input2")

output = tf.add_n([input1, input2], "add")

writer = tf.summary.FileWriter("../saved_model/log", tf.get_default_graph())
writer.close()