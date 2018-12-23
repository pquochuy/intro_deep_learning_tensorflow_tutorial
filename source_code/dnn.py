"""
A DNN (a.k.a Multilayer Perceptron) with 2 hidden fully connected layers.
"""
import tensorflow as tf
import numpy as np

class DNN(object):
    def __init__(self, config):
        self.config = config
        # place holder for input feature vectors and one-hot encoding output
        self.X = tf.placeholder("float", shape=[None, self.config.num_input], name='X')
        self.y = tf.placeholder("float", shape=[None, self.config.num_classes], name='y')
        # place holder for dropout
        self.dropout_keep_prob = tf.placeholder("float", name="dropout_keep_prob")

        self.construct()

    def construct(self):
        # layers weight & bias
        self.w = {
            'h1': tf.Variable(tf.random_normal([self.config.num_input, self.config.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.config.n_hidden_1, self.config.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.config.n_hidden_2, self.config.num_classes]))
        }
        self.b = {
            'h1': tf.Variable(tf.random_normal([self.config.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.config.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.config.num_classes]))
        }

        #with tf.device('/gpu:0'), tf.variable_scope("fully-connected-layers"):
        with tf.device('/cpu:0'), tf.variable_scope("fully-connected-layers"):
            h1 = tf.add(tf.matmul(self.X, self.w['h1']), self.b['h1'])
            h1_relu = tf.nn.relu(h1)
            h1_dropout = tf.nn.dropout(h1_relu, self.dropout_keep_prob)

            h2 = tf.add(tf.matmul(h1_dropout, self.w['h2']), self.b['h2'])
            h2_relu = tf.nn.relu(h2)
            h2_dropout = tf.nn.dropout(h2_relu, self.dropout_keep_prob)

        # network's output
        with tf.device('/cpu:0'), tf.variable_scope("output"):
            self.output = tf.add(tf.matmul(h2_dropout, self.w['out']), self.b['out']) # logit
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
