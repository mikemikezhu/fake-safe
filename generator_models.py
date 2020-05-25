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


""" Text -> State """


class TextEncoderGeneratorModelCreator(AbstractModelCreator):

    def __init__(self, vocabulary_size, max_sequence_length):
        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = max_sequence_length

    def create_model(self):

        # Text model to extract sequence features using RNN
        text_input = Input(shape=(self.max_sequence_length))
        text_embedding = Embedding(input_dim=self.vocabulary_size,
                                   output_dim=100,
                                   mask_zero=True)(text_input)

        # Return the hidden state of the final time stamp of RNN
        text_rnn = GRU(256, return_state=True)
        _, state_h = text_rnn(text_embedding)

        # Create model
        model = Model(inputs=text_input, outputs=state_h)

        print('Text to State Generator model:')
        model.summary()

        return model


""" State -> Image """


class StateEncoderGeneratorModelCreator(AbstractModelCreator):

    def __init__(self, output_shape):
        self.output_shape = output_shape

    def create_model(self):

        # Generate image using text model hidden state
        state_input = Input(shape=(256,))

        image_dense_1 = Dense(256)(state_input)
        image_relu_1 = LeakyReLU(alpha=0.2)(image_dense_1)
        image_norm_1 = BatchNormalization(momentum=0.8)(image_relu_1)

        image_dense_2 = Dense(512)(image_norm_1)
        image_relu_2 = LeakyReLU(alpha=0.2)(image_dense_2)
        image_norm_2 = BatchNormalization(momentum=0.8)(image_relu_2)

        image_dense_3 = Dense(1024)(image_norm_2)
        image_relu_3 = LeakyReLU(alpha=0.2)(image_dense_3)
        image_norm_3 = BatchNormalization(momentum=0.8)(image_relu_3)

        image_dense_output = Dense(np.prod(self.output_shape),
                                   activation='tanh')(image_norm_3)
        image_output = Reshape(self.output_shape)(image_dense_output)

        model = Model(inputs=state_input, outputs=image_output)

        print('State to Image Generator model:')
        model.summary()

        return model


""" Image -> State """


class StateDecoderGeneratorModelCreator(AbstractModelCreator):

    def __init__(self, input_shape):
        self.input_shape = input_shape

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

        model.add(Dense(256, activation='linear'))  # Hidden state

        print('Image to State Generator model:')
        model.summary()

        return model


""" State -> Text """


class TextDecoderGeneratorModelCreator(AbstractModelCreator):

    def __init__(self, vocabulary_size, max_sequence_length):
        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = max_sequence_length

    def create_model(self):

        state_input = Input(shape=(256,))  # Hidden state

        # Text model to extract sequence features
        text_input = Input(shape=(self.max_sequence_length))
        text_embedding = Embedding(input_dim=self.vocabulary_size,
                                   output_dim=100,
                                   mask_zero=True)(text_input)

        text_rnn = GRU(256)
        text_gru_output = text_rnn(text_embedding,
                                   initial_state=state_input)

        text_dense_output = Dense(self.vocabulary_size,
                                  activation='softmax')
        text_output = text_dense_output(text_gru_output)

        # Create model
        model = Model(inputs=[state_input, text_input],
                      outputs=text_output)

        optimizer = Adam(0.0002, 0.5)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        print('State to Text Generator model:')
        model.summary()

        return model
