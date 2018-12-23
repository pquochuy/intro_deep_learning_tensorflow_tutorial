"""
An RNN with Tensorflow.
"""
import tensorflow as tf
import numpy as np

class RNN(object):
    def __init__(self, config):
        self.config = config
        # place holder for input feature vectors and one-hot encoding output
        self.X = tf.placeholder("float",
                                shape=[None, self.config.timesteps, self.config.num_input],
                                name='X')
        self.y = tf.placeholder("float", shape=[None, self.config.num_classes], name='y')
        # place holder for dropout
        self.dropout_keep_prob = tf.placeholder("float", name="dropout_keep_prob")

        self.seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN

        self.construct()

    def construct(self):
        # Define weights
        self.w = {
            'out': tf.Variable(tf.random_normal([self.config.n_hidden, self.config.num_classes]))
        }
        self.b = {
            'out': tf.Variable(tf.random_normal([self.config.num_classes]))
        }

        with tf.device('/cpu:0'), tf.name_scope("recurrent_layer"):
            # Define a lstm cell with tensorflow
            #rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden, forget_bias=1.0)
            rnn_cell = tf.contrib.rnn.GRUCell(self.config.n_hidden)
            # Get RNN cell output
            #outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, self.X, dtype=tf.float32)
            outputs, states = tf.nn.dynamic_rnn(rnn_cell, self.X, sequence_length=self.seq_len, dtype=tf.float32)

            last_rnn_output = outputs[:, -1]
            last_rnn_output = tf.nn.dropout(last_rnn_output, self.dropout_keep_prob)

        # network's output
        with tf.device('/cpu:0'), tf.variable_scope("output"):
            self.output = tf.add(tf.matmul(last_rnn_output, self.w['out']), self.b['out']) # logit
            self.y_hat = tf.argmax(self.output, 1, name='y_hat') # predicted labels

        # network's losses
        with tf.device('/cpu:0'), tf.name_scope("loss"):
            # cross-entropy loss
            self.output_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.output)
            self.output_loss = tf.reduce_mean(self.output_loss) # summing over all samples of the batch

            # add on regularization
            l2_loss = self.config.l2_reg_lambda * \
                      sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

            # total loss
            self.loss = self.output_loss + l2_loss

        # calculate accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.y_hat, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
