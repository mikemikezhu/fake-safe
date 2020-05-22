import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import add, BatchNormalization, Dense, Embedding, Flatten, GlobalMaxPool1D, Input, LeakyReLU, GRU, Lambda, Reshape
from abc import ABC, abstractmethod

import tensorflow as tf
print('Tensorflow version: {}'.format(tf.__version__))


"""
Abstract Model Creator
"""


class AbstractModelCreator(ABC):

    @abstractmethod
    def create_model(self):
        raise NotImplementedError('Abstract class shall not be implemented')


"""
Generator Model Creator
"""


class Image2ImageGeneratorModelCreator(AbstractModelCreator):

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


class Text2ImageGeneratorModelCreator(AbstractModelCreator):

    def __init__(self, vocabulary_size, max_sequence_length, output_shape):
        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = max_sequence_length
        self.output_shape = output_shape

    def create_model(self):

        model = Sequential()

        # The model must be informed that some part of the data is
        # padding and should be ignored by using masking
        # See: https://www.tensorflow.org/guide/keras/masking_and_padding#masking
        model.add(Embedding(input_dim=self.vocabulary_size,
                            output_dim=100,
                            input_length=self.max_sequence_length,
                            mask_zero=True))
        model.add(GlobalMaxPool1D())

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.output_shape), activation='tanh'))
        model.add(Reshape(self.output_shape))

        print('Text to Image Generator model:')
        model.summary()

        return model


class Image2TextGeneratorModelCreator(AbstractModelCreator):

    def __init__(self, input_shape, vocabulary_size, max_sequence_length):
        self.input_shape = input_shape
        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = max_sequence_length

    def create_model(self):

        # Image model to extract image features
        image_input = Input(shape=self.input_shape)
        image_flatten = Flatten()(image_input)

        image_dense_1 = Dense(512)(image_flatten)
        image_relu_1 = LeakyReLU(0.2)(image_dense_1)

        image_dense_2 = Dense(256)(image_relu_1)
        image_relu_2 = LeakyReLU(0.2)(image_dense_2)

        # Text model to extract sequence features
        text_input = Input(shape=(self.max_sequence_length))
        text_embedding = Embedding(input_dim=self.vocabulary_size,
                                   output_dim=100,
                                   mask_zero=True)(text_input)
        text_lstm = GRU(256)(text_embedding)

        # Merge the two input models
        merged_input = add([image_relu_2, text_lstm])
        merged_dense = Dense(256)(merged_input)
        merged_relu = LeakyReLU(0.2)(merged_dense)

        outputs = Dense(self.vocabulary_size,
                        activation='softmax')(merged_relu)
        model = Model(inputs=[image_input, text_input],
                      outputs=outputs)

        optimizer = Adam(0.0002, 0.5)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        print('Image to Text Generator model:')
        model.summary()

        return model


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
