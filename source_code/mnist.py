# conda install matplotlib


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import random

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


train_X = mnist.train.images
train_y = mnist.train.labels
test_X = mnist.test.images
test_y = mnist.test.labels

N = 5
f, axarr = plt.subplots(N,N)
for i in range(N):
    for j in range(N):
        rand_ind = random.randint(0,len(train_X))
        X = np.array(train_X[rand_ind], dtype='float')
        X = X.reshape((28, 28))
        axarr[i,j].imshow(X, cmap='gray')
plt.show()
