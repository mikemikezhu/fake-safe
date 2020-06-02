from abstract_models import AbstractModelCreator
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU

import tensorflow as tf
print('Tensorflow version: {}'.format(tf.__version__))


"""
Classifier Model Creator
"""


class ClassifierModelCreator(AbstractModelCreator):

    def __init__(self, input_shape, output_classes, name):
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.name = name

    def create_model(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.input_shape))

        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(self.output_classes,
                        activation='softmax'))

        optimizer = Adam(0.0002, 0.5)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        print('Classifier model:')
        model.summary()

        return model
