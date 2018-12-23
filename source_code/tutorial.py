from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
from datetime import datetime

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

from dnn_config import DNN_Config
from dnn import DNN
#from cnn_config import CNN_Config
#from cnn import CNN
#from rnn_config import RNN_Config
#from rnn import RNN


tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.app.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # python3
    print("{}={}".format(attr.upper(), value))
print("")

# path where some output and checkpoints are stored
out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

# Configuration
config = DNN_Config()
# config = CNN_Config()
# config = RNN_Config()

# Load MNIST dataset
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Trainging Parameters
learning_rate = 1e-3
num_training_step = 1000
batch_size = 128
display_every = 10
evaluate_every = 10

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        # Network construction
        net = DNN(config=config)
        # net = CNN(config=config)
        # net = RNN(config=config)

        # Define training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(net.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # model saver
        saver = tf.train.Saver(tf.all_variables(), max_to_keep = 5)

        # initialize all variables
        print("Model initialized")
        sess.run(tf.global_variables_initializer())

        # training step
        def train(x_batch, y_batch):

            # Can you guess why we need this line?
            #seq_len = np.ones(len(x_batch),dtype=int) * config.timesteps

            feed_dict = {
                net.X: x_batch,
                net.y: y_batch,
                net.dropout_keep_prob: config.dropout_keep_prob,
                #net.seq_len: seq_len # Can you guess why we need this line?
            }
            _, step, loss, acc = sess.run([train_op, global_step, net.loss, net.accuracy], feed_dict)
            return step, loss, acc

        # testing step
        def eval(x_batch, y_batch):

            # Can you guess why we need this line?
            #seq_len = np.ones(len(x_batch),dtype=int) * config.timesteps

            feed_dict = {
                net.X: x_batch,
                net.y: y_batch,
                net.dropout_keep_prob: 1.0,
                #net.seq_len: seq_len  # Can you guess why we need this line?
            }
            _, loss, yhat, acc = sess.run(
                [global_step, net.loss, net.y_hat, net.accuracy],
                feed_dict)
            return loss, acc, yhat

        for step in range(1, num_training_step + 1):
            x_batch, y_batch = mnist.train.next_batch(batch_size)

            # Can you guess why we need this line?
            # x_batch = np.reshape(x_batch, (-1, config.input_height, config.input_width, config.input_channel))

            # Can you guess why we need this line?
            # reshape to (batch_size, num_input)
            # x_batch = np.reshape(x_batch, (-1, config.timesteps, config.num_input))

            train_step, train_loss, train_acc = train(x_batch, y_batch)
            # modal evaluation
            if step % display_every == 0:
                #time_str = datetime.now().isoformat()
                print("Step {}, loss {:.4f}, accuracy {:.3f}".format(train_step, train_loss, train_acc))

            # modal evaluation
            if step % evaluate_every == 0:
                test_X = mnist.test.images

                # Can you guess why we need this line?
                # test_X = np.reshape(test_X, (-1, config.input_height, config.input_width, config.input_channel))

                # Can you guess why we need this line?
                # reshape to (batch_size, num_input)
                # test_X = np.reshape(test_X, (-1, config.timesteps, config.num_input))

                test_y = mnist.test.labels
                test_loss, test_acc, test_yhat = eval(test_X, test_y)
                print("Evaluation: loss {:.4f}, accuracy {:.3f}".format(test_loss, test_acc))

                # save the current model
                checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(step))
                save_path = saver.save(sess, checkpoint_name)
