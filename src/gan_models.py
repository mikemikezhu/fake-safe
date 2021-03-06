import tensorflow as tf
from tensorflow.keras.models import Sequential
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
Decoder GAN Model Creator
This is a logical model to combine encoder generator and decoder generator
"""


class DecoderGanModelCreator(AbstractModelCreator):

    def __init__(self,
                 encoder_generator,
                 decoder_generator,
                 loss='mae'):
        self.encoder_generator = encoder_generator
        self.decoder_generator = decoder_generator
        self.loss = loss

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
        model.compile(loss=self.loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        print('Decoder GAN model:')
        model.summary()

        return model
