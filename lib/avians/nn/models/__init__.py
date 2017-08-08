
# Uncomment below for debugging
# import theano
# theano.config.exception_verbosity = "high"
# theano.config.optimizer = "None"

import sys
sys.setrecursionlimit(10000)

import keras.backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Convolution1D, MaxPooling1D
from keras.layers import Input, Merge, merge
from keras.layers.core import Dense, Flatten, Reshape, Dropout, Activation, TimeDistributedDense, Permute, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import cv2

import os

import avians.nn.common as anc
import avians.util.image as aui
import avians.util.arabic as aua
import avians.util.arabic_defs as auad


import logging
LOG = logging.getLogger(__name__)

