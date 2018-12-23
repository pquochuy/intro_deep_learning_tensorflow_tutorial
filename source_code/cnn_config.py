'''
Holding CNN parameters
'''

class CNN_Config(object):
    def __init__(self):

        self.input_height = 28 # MNIST data input (img shape: 28*28*1)
        self.input_width = 28 # MNIST data input (img shape: 28*28*1)
        self.input_channel = 1 # MNIST data input (img shape: 28*28*1)
        self.num_classes = 10 # MNIST total classes (0-9 digits)

        self.dropout_keep_prob = 0.9
        self.l2_reg_lambda = 1e-4
