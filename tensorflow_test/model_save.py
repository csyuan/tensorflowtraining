#coding:utf-8

import numpy as np
import tensorflow as tf

# w1 = tf.Variable(tf.random_normal([2,3],stddev = 1, seed=1))
# w2 = tf.Variable(tf.random_normal([2,2],stddev = 1, seed=1))

# v1 = tf.Variable(1.0, name = "v1")
# v2 = tf.Variable(2.0, name = "v2")

# result = v1 + v2
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")
result = v1 + v2
saver = tf.train.Saver({"v1": v1, "v2": v2})

# saver = tf.train.import_meta_graph("saved_model/model.ckpt.meta")

sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()
# saver.save(sess, "saved_model/model.ckpt")

saver.restore(sess, "saved_model/model.ckpt")
# print sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))
print sess.run(result)
print "----------------------"
# reader = tf.train.NewCheckpointReader("saved_model/model.ckpt")
# all_variables = reader.get_variable_to_shape_map()
# for variable_name in all_variables:
#     print variable_name, all_variables[variable_name]
# print "value for variable v1 is ", reader.get_tensor('v1')

sess.close()

