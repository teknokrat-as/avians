
from keras.datasets import mnist
import numpy as np

from . import *
import component_autoencoder_v1 as cav1

def test_train_model(): 
    (x_train, _), (x_test, _) = mnist.load_data()

    # We will normalize all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784.

    LOG.debug("=== x_train.shape ===")
    LOG.debug(x_train.shape)
    
    # x_train = x_train.astype('float32') / 255.
    # x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
    x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    
    cav1.train_model(x_train, validation_data=x_test, validation_split=0)
