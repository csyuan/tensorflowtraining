import tensorflow as tf
import numpy as np

lstm_hidden_size = 100
batch_size = 10

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size)
state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

loss = 0.0
num_steps = 10

for i in range(num_steps):
    if i > 0:
        tf.get_variable_scope().reuse_variables()
    # lstm_output, state = lstm(current_input, state)