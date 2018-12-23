'''
Holding DNN parameters
'''

class RNN_Config(object):
    def __init__(self):

        self.n_hidden = 256 # 1st layer number of neurons
        self.num_input = 28 # MNIST data input (img shape: 28*28)
        self.timesteps = 28
        self.num_classes = 10 # MNIST total classes (0-9 digits)

        self.dropout_keep_prob = 0.9
        self.l2_reg_lambda = 1e-4
