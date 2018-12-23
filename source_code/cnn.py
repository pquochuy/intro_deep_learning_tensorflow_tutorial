"""
A CNN with Tensorflow.
"""
import tensorflow as tf
import numpy as np

class CNN(object):
    def __init__(self, config):
        self.config = config
        # place holder for input feature vectors and one-hot encoding output
        self.X = tf.placeholder("float",
                                shape=[None,
                                       self.config.input_height,
                                       self.config.input_width,
                                       self.config.input_channel],
                                name='X')
        self.y = tf.placeholder("float", shape=[None, self.config.num_classes], name='y')
        # place holder for dropout
        self.dropout_keep_prob = tf.placeholder("float", name="dropout_keep_prob")

        self.construct()

    # Create some wrappers for simplicity
    def conv2d(self, X, W, b, stride = 1): # conv2d wrapper
        # Conv2D wrapper, with bias and relu activation
        conv = tf.nn.conv2d(X, W, strides = [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, b)
        conv = tf.nn.relu(conv)
        return conv

    def maxpool2d(self, X, k = 2): # maxpool2d wrapper
        return tf.nn.max_pool(X, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def construct(self):
        # layers weight & bias
        self.w = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, self.config.num_classes]))
        }

        self.b = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.config.num_classes]))
        }

        # Convolutional layers
        with tf.device('/cpu:0'), tf.variable_scope("conv-layers"):
            # Conv layer 1
            conv1 = self.conv2d(self.X, self.w['wc1'], self.b['bc1'])
            # Max Pooling (down-sampling)
            conv1_pool = self.maxpool2d(conv1, k = 2)
            conv1_dropout = tf.nn.dropout(conv1_pool, self.dropout_keep_prob)

            # Conv Layer 2
            conv2 = self.conv2d(conv1_dropout, self.w['wc2'], self.b['bc2'])
            # Max Pooling (down-sampling)
            conv2_pool = self.maxpool2d(conv2, k = 2)
            conv2_dropout = tf.nn.dropout(conv2_pool, self.dropout_keep_prob)

        # fully connected layer
        with tf.device('/cpu:0'), tf.variable_scope("fully-connected-layers"):
            # flatten conv feature map
            flattened = tf.reshape(conv2_dropout, [-1, self.w['wd1'].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(flattened, self.w['wd1']), self.b['bd1'])
            fc1_relu = tf.nn.relu(fc1)
            fc1_dropout = tf.nn.dropout(fc1_relu, self.dropout_keep_prob)

        # network's output
        with tf.device('/cpu:0'), tf.variable_scope("output"):
            self.output = tf.add(tf.matmul(fc1_dropout, self.w['out']), self.b['out']) # logit
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
