from abstract_models import AbstractModelCreator
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import add, BatchNormalization, Dense, Dropout, Embedding, Flatten, GlobalMaxPool1D, Input, LeakyReLU, GRU, Lambda, Reshape

import tensorflow as tf
print('Tensorflow version: {}'.format(tf.__version__))


"""
Generator Model Creator
"""

""" Image -> Image """


class ImageGeneratorModelCreator(AbstractModelCreator):

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def create_model(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.input_shape))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.input_shape), activation='tanh'))
        model.add(Reshape(self.input_shape))

        print('Image to Image Generator model:')
        model.summary()

        return model


""" Word -> Image """


class TextEncoderGeneratorModelCreator(AbstractModelCreator):

    def __init__(self, input_shape, vocabulary_size):
        self.input_shape = input_shape
        self.vocabulary_size = vocabulary_size

    def create_model(self):

        model = Sequential()

        model.add(Dense(64, input_shape=(1,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.input_shape), activation='tanh'))
        model.add(Reshape(self.input_shape))

        print('Text to Image Generator model:')
        model.summary()

        return model


""" Image -> Word """


class TextDecoderGeneratorModelCreator(AbstractModelCreator):

    def __init__(self, input_shape, vocabulary_size):
        self.input_shape = input_shape
        self.vocabulary_size = vocabulary_size

    def create_model(self):

        # Image model to extract image features
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.vocabulary_size,
                        activation='softmax'))

        print('Image to Text Generator model:')
        model.summary()

        return model
