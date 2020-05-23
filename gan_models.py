import tensorflow as tf
from tensorflow.keras.layers import add, BatchNormalization, Dense, Embedding, Flatten, GlobalMaxPool1D, Input, LeakyReLU, GRU, Lambda, Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from abstract_models import AbstractModelCreator


"""
Encoder GAN Model Creator
This is a logical model to combine encoder generator and encoder discriminator
"""


class EncoderGanModelCreator(AbstractModelCreator):

    def __init__(self,
                 encoder_generator,
                 encoder_discriminator):
        self.encoder_generator = encoder_generator
        self.encoder_discriminator = encoder_discriminator

    def create_model(self):

        # 1) Set generator to trainable
        self.encoder_generator.trainable = True
        # 2) Set discriminator to non-trainable
        self.encoder_discriminator.trainable = False

        # Create logical model to combine encoder generator and encoder discriminator
        model = Sequential()

        model.add(self.encoder_generator)
        model.add(self.encoder_discriminator)

        optimizer = Adam(0.0002, 0.5)
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        print('Encoder GAN model:')
        model.summary()

        return model


"""
Text Encoder GAN Model Creator
This is a logical model to combine text and state encoder generator and encoder discriminator
"""


class TextEncoderGanModelCreator(AbstractModelCreator):

    def __init__(self,
                 text_encoder_generator,
                 state_encoder_generator,
                 encoder_discriminator):
        self.text_encoder_generator = text_encoder_generator
        self.state_encoder_generator = state_encoder_generator
        self.encoder_discriminator = encoder_discriminator

    def create_model(self):

        # 1) Set generator to trainable
        self.text_encoder_generator.trainable = True
        self.state_encoder_generator.trainable = True
        # 2) Set discriminator to non-trainable
        self.encoder_discriminator.trainable = False

        # Create logical model to combine encoder generator and encoder discriminator
        model = Sequential()

        model.add(self.text_encoder_generator)
        model.add(self.state_encoder_generator)
        model.add(self.encoder_discriminator)

        optimizer = Adam(0.0002, 0.5)
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        print('Encoder GAN model:')
        model.summary()

        return model


print('Tensorflow version: {}'.format(tf.__version__))

"""
Decoder GAN Model Creator
This is a logical model to combine encoder generator and decoder generator
"""


class DecoderGanModelCreator(AbstractModelCreator):

    def __init__(self,
                 encoder_generator,
                 decoder_generator):
        self.encoder_generator = encoder_generator
        self.decoder_generator = decoder_generator

    def create_model(self):

        # 1) Set encoder generator to non-trainable
        self.encoder_generator.trainable = False
        # 2) Set decoder generator to trainable
        self.decoder_generator.trainable = True

        # Create logical model to combine encoder generator and decoder generator
        model = Sequential()

        model.add(self.encoder_generator)
        model.add(self.decoder_generator)

        optimizer = Adam(0.0002, 0.5)
        model.compile(loss='mae',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        print('Decoder GAN model:')
        model.summary()

        return model
