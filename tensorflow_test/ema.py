import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")

for variables in tf.global_variables():
    print variables.name

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
for variables in tf.global_variables():
    print variables.name

print ema.variables_to_restore()

# saver = tf.train.Saver()
# saver = tf.train.Saver({"v/ExponentialMovingAverage" : v})
saver = tf.train.Saver(ema.variables_to_restore())

with tf.Session() as sess:
    # init_op = tf.global_variables_initializer()
    # sess.run(init_op)
    # sess.run(tf.assign(v, 10))
    # sess.run(maintain_averages_op)
    # saver.save(sess,"../saved_model/model.ckpt")
    # print sess.run([v, ema.average(v)])
    saver.restore(sess, "../saved_model/model.ckpt")
    print sess.run(v)