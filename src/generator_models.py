from abstract_models import AbstractModelCreator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, LeakyReLU, Reshape

import tensorflow as tf
print('Tensorflow version: {}'.format(tf.__version__))


"""
Generator Model Creator
"""


class GeneratorModelCreator(AbstractModelCreator):

    def __init__(self, input_shape, output_shape,
                 from_image=False, to_image=False,
                 activation='linear'):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.from_image = from_image
        self.to_image = to_image
        self.activation = activation

    def create_model(self):

        model = Sequential()

        if self.from_image:
            model.add(Flatten(input_shape=self.input_shape))
        else:
            model.add(Dense(64, input_shape=self.input_shape))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dropout(0.2))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.2))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.2))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        if self.to_image:
            model.add(Dense(np.prod(self.output_shape),
                            activation=self.activation))
            model.add(Reshape(self.output_shape))
        else:
            model.add(Dense(self.output_shape,
                            activation=self.activation))

        print('Generator model:')
        model.summary()

        return model
