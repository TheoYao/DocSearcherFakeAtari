# encoding: utf-8

import pickle
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import pooling
from keras.layers.convolutional import Conv2D


class CnnLstm(object):
    def __init__(cls):
        in_data = cls.load_data()

    def create_network(cls):
        model = Sequential()
        model.add(Conv2D([1, 30, 1 ,32], 32,
                         strides=[1, 1, 1, 1],
                         input_shape

    @staticmethod
    def load_data(path=''):
        if not path:
            path = '../Data/kmdata_match_train.pkl'
        with open(path, 'rb') as in_file:
            data = pickle.load(in_file)
        return data
