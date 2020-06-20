from abstract_models import AbstractModelCreator
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU

import tensorflow as tf
print('Tensorflow version: {}'.format(tf.__version__))


"""
Discriminator Model Creator
"""


class DiscriminatorModelCreator(AbstractModelCreator):

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def create_model(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.input_shape))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(0.0002, 0.5)
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        print('Discriminator model:')
        model.summary()

        return model