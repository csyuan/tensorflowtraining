# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from data.Data import Data


class RNN:
    inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputs")
    outs = tf.placeholder(dtype=tf.float32, shape=[None, None], name="outs")
    keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
    learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
    sess = tf.Session()

    _loss = None
    _minimize_loss = None
    _predict = None
    _correct_prediction = None

    def __init__(self, vocab_size=4, word_dimension=32, class_num=2, unit_num=32, layer_num=1):
        self.vocab_size = vocab_size
        self.word_dimension = word_dimension
        self.class_num = class_num
        self.unit_num = unit_num
        self.layer_num = layer_num
        with tf.name_scope('word_embedding'):
            word_embedding = tf.get_variable(name="word_embedding",
                                             dtype=tf.float32,
                                             shape=[self.vocab_size, self.word_dimension])
            tf.summary.histogram('word_embedding', word_embedding)
        with tf.name_scope("look_up"):
            look_up = tf.nn.embedding_lookup(name="look_up", params=word_embedding, ids=self.inputs)
            tf.summary.histogram("look_up", look_up)

        with tf.name_scope("rnn_unit"):
            cell = tf.contrib.rnn.GRUCell(self.unit_num)  # , state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=self.keep_prob)
            if self.layer_num > 1:
                cell = tf.contrib.rnn.MultiRNNCell([cell] * self.layer_num, state_is_tuple=True)

        with tf.name_scope("rnn"):
            outs, states = tf.nn.dynamic_rnn(cell=cell, inputs=look_up, dtype=tf.float32)

        # 从不同长度抓取特
        with tf.name_scope("sss"):
            filters = tf.get_variable("filters", shape=[1, self.unit_num, self.unit_num],
                                      dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=116))
            weights = tf.nn.conv1d(outs, filters=filters, stride=1, padding="SAME")
            weights = tf.nn.softmax(weights, dim=1)
            outs = tf.reduce_sum(outs * weights, axis=1)

        outs = tf.nn.dropout(outs, keep_prob=self.keep_prob, name="dropout")
        with tf.name_scope('h2y_weights'):
            h2y_weights = tf.Variable(tf.truncated_normal([self.unit_num, self.class_num],
                                                          stddev=0.1,
                                                          seed=10,
                                                          dtype=tf.float32))
            tf.summary.histogram("h2y_weights", h2y_weights)

        with tf.name_scope('h2y_weights'):
            h2y_biases = tf.get_variable(name="h2y_biases",
                                         shape=[self.class_num],
                                         dtype=tf.float32)
            tf.summary.histogram("h2y_biases", h2y_biases)

        self.merged = tf.summary.merge_all()

        with tf.name_scope('predicts'):
            self.__outs = tf.matmul(outs, h2y_weights) + h2y_biases

    def train(self):
        _predict = self.__outs
        with tf.name_scope('loss_train'):
            _loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_predict, labels=self.outs))
            tf.summary.scalar("loss_train", _loss)

        alg = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        _minimize_loss = alg.minimize(loss=_loss, name="minimize_loss")

        self._loss = _loss
        self._minimize_loss = _minimize_loss
        self._predict = _predict

        return _loss, _minimize_loss, _predict

    def predict(self, fed_dict):
        fed_dict[self.keep_prob] = 1.0
        __predicts = self.sess.run(self.__outs, feed_dict=fed_dict)
        return __predicts

    def evaluate(self):
        pass

    def load_model(self):
        # noinspection PyBroadException
        try:
            tf.train.Saver().restore(self.sess, "./model_file/model.sd")
        except Exception, ex:
            # print ex
            self.sess.run(tf.global_variables_initializer())

    def save_model(self):
        saver = tf.train.Saver()
        saver_def = saver.as_saver_def()
        print saver_def.filename_tensor_name
        print saver_def.restore_op_name

        saver.save(self.sess, "./model_file/model.sd")
        tf.train.write_graph(self.sess.graph_def, "./model_file", "model.proto", as_text=False)
        tf.train.write_graph(self.sess.graph_def, "./model_file", "model.txt", as_text=True)

    def train_writer(self):
        return tf.summary.FileWriter('log/train',
                                      self.sess.graph)

    def train_min_data(self, min_inputs, min_labels, keep_prob=0.5, learning_rate=0.1):
        _, loss_total = self.sess.run([self._minimize_loss, self._loss],
                                      feed_dict={self.inputs: min_inputs,
                                                 self.outs: min_labels,
                                                 self.keep_prob: keep_prob,
                                                 self.learning_rate: learning_rate})

        return loss_total

    def test_min_data(self, min_inputs, min_labels):
        preidcts, loss = self.sess.run([self._predict, self._loss],
                                       feed_dict={self.inputs: min_inputs,
                                                  self.outs: min_labels,
                                                  self.keep_prob: 1.0})
        rightnum = 0
        if isinstance(preidcts, np.ndarray) and isinstance(min_labels, np.ndarray):
            predict_id = np.argmax(preidcts, 1)
            target_id = np.argmax(min_labels, 1)

            confusion_matrix = {}
            for i in range(0, predict_id.__len__()):
                key = str(target_id[i]) + "->" + str(predict_id[i])

                if target_id[i] == predict_id[i]:
                    rightnum += 1

                if confusion_matrix.__contains__(key):
                    confusion_matrix[key] += 1
                else:
                    confusion_matrix[key] = 1

            return confusion_matrix, loss, rightnum, predict_id.__len__()
        else:
            return None, None, None, None

    def test_datas(self, min_inputs, min_labels):
        confusion_matrix = {}
        if isinstance(min_labels, list) and isinstance(min_inputs, list):
            total_test_loss = 0
            total_test_right_num = 0
            total_nums = 0
            for i in range(0, min_labels.__len__()):
                hm, loss, right_num, total_num = self.test_min_data(min_inputs[i], min_labels[i])
                total_test_loss += loss
                total_test_right_num += right_num
                total_nums += total_num

                if isinstance(hm, dict):
                    for key, value in hm.items():
                        if confusion_matrix.__contains__(key):
                            confusion_matrix[key] += value
                        else:
                            confusion_matrix[key] = value
            avg_loss = total_test_loss / min_labels.__len__()
            right_rate=(total_test_right_num * 1.0 / total_nums)
            print "-------test_avg_loss and test_right_rate----------------"
            print avg_loss, right_rate
            print "-------test_avg_loss and test_right_rate----------------"
            print "-------confusion_matrix-------------"
            print confusion_matrix
            print "-------confusion_matrix-------------"

            return confusion_matrix, avg_loss,right_rate

        else:
            return None, None,None


def train_model():
    data = Data("../data/trainData.json", "../data/dataDict/")
    train_inputs = data.train_input_datas
    train_labels = data.train_label_datas
    test_inputs = data.test_input_datas
    test_labels = data.test_label_datas
    valid_inputs = data.valid_input_datas
    valid_labels = data.valid_label_datas

    print valid_labels.__len__()
    print valid_inputs.__len__()

    batch_size = 500

    indexes = np.arange(train_inputs.__len__())

    rnn_obj = RNN(vocab_size=data.id2Word.__len__(), class_num=data.class2Id.__len__())

    rnn_obj.train()

    train_writer = rnn_obj.train_writer()
    rnn_obj.load_model()
    # rnn_obj.test_datas(test_inputs, test_labels)
    # rnn_obj.save_model()
    train_loss = []
    valid_loss = []
    valid_right_rates=[]
    plt.ion()

    for step in range(0, 10):
        np.random.shuffle(indexes)

        loss_total = 0
        word_num = 0
        start_time = time.time()

        for i in indexes:
            min_inputs = train_inputs[i]
            min_labels = train_labels[i]

            current_loss = rnn_obj.train_min_data(min_inputs, min_labels)

            word_num += min_inputs.size
            loss_total += current_loss

            if i % batch_size == 0:
                end_time = time.time()
                loss_total = loss_total / batch_size

                # if train_loss.__len__()>0:
                #     loss_total=0.999*train_loss[-1]+0.001*loss_total

                print "-------train_data------------"
                print loss_total
                print "speed:word numbers/s:" + str(word_num * 1.0 / (end_time - start_time))
                print "-----------valid_data-------------"
                _, valid_avg_loss,valid_right_rate = rnn_obj.test_datas(valid_inputs, valid_labels)
                valid_loss.append(valid_avg_loss)
                train_loss.append(loss_total)
                valid_right_rates.append(valid_right_rate)
                step += 1
                loss_total = 0
                word_num = 0
                start_time = time.time()
                rnn_obj.save_model()

                plt.plot(train_loss, 'cx-', valid_loss, 'mo:')#, valid_right_rates, 'kp-.');

                plt.pause(0.1)
        print "-----------test_data-------------"
        rnn_obj.test_datas(test_inputs, test_labels)

        rnn_obj.save_model()


if __name__ == '__main__':
    # data = Data("../data/trainData.json", "../data/dataDict/")
    # train_inputs = data.train_input_datas
    # train_labels = data.train_label_datas
    # test_inputs = data.test_input_datas
    # test_labels = data.test_label_datas
    # valid_inputs = data.valid_input_datas
    # valid_labels = data.valid_label_datas
    #
    # indexes = np.arange(train_inputs.__len__())
    #
    # rnn_obj = RNN(vocab_size=data.id2Word.__len__(), class_num=data.class2Id.__len__())
    #
    # rnn_obj.train()
    #
    # train_writer = rnn_obj.train_writer()
    # rnn_obj.load_model()
    # feed = {}
    # feed[rnn_obj.inputs] = np.array([[115,116,193,179,1275,391,304,51,46,382,598,75,811,631,142,210]])
    # a = rnn_obj.predict(feed)
    # print a
    train_model()
    # data = Data("../data/trainData.json", "../data/dataDict/")
