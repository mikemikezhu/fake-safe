from abstract_models import AbstractModelCreator
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import add, BatchNormalization, Dense, Dropout, Embedding, Flatten, GlobalMaxPool1D, Input, LeakyReLU, LSTM, GRU, TimeDistributed, Lambda, Reshape

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


""" Train: Text -> State -> Text """


class TextTrainModelCreator(AbstractModelCreator):

    def __init__(self, vocabulary_size, max_sequence_length):
        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = max_sequence_length

    def create_model(self):
        """ Create text generator train model """
        # Text generator train model includes encoder and decoder
        # Reference:
        # https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
        encoder_inputs = Input(shape=(self.max_sequence_length,))
        encoder_embedding = Embedding(self.vocabulary_size,
                                      256,
                                      mask_zero=True)
        encoder_inputs_x = encoder_embedding(encoder_inputs)
        # Discard the output of the encoder RNN, only focus on the states
        encoder_lstm = GRU(256, return_state=True)
        encoder_outputs, state_h = encoder_lstm(encoder_inputs_x)
        encoder_states = state_h

        # Set up the decoder, using "encoder_states" as initial state.
        decoder_inputs = Input(shape=(self.max_sequence_length,))
        decoder_embedding = Embedding(self.vocabulary_size,
                                      256,
                                      mask_zero=True)
        decoder_inputs_x = decoder_embedding(decoder_inputs)
        decoder_lstm = GRU(256,
                           return_sequences=True,
                           return_state=True)
        decoder_outputs, _ = decoder_lstm(decoder_inputs_x,
                                          initial_state=encoder_states)
        decoder_dense = TimeDistributed(Dense(self.vocabulary_size,
                                              activation='softmax'))
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # "encoder_input_data" & "decoder_input_data" into "decoder_target_data"
        train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        train_model.compile(optimizer='rmsprop',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        return train_model, encoder_embedding, encoder_lstm, decoder_embedding, decoder_lstm, decoder_dense


""" Inference: Text -> State """


class TextEncoderGeneratorModelCreator(AbstractModelCreator):

    def __init__(self,
                 encoder_embedding,
                 encoder_lstm,
                 max_sequence_length):
        self.encoder_embedding = encoder_embedding
        self.encoder_lstm = encoder_lstm
        self.max_sequence_length = max_sequence_length

    def create_model(self):
        """ Create text inference encoder model """
        # The encoder model remains the same
        encoder_inputs = Input(shape=(self.max_sequence_length,))
        encoder_inputs_x = self.encoder_embedding(encoder_inputs)
        encoder_outputs, state_h = self.encoder_lstm(encoder_inputs_x)
        encoder_states = state_h
        encoder_model = Model(encoder_inputs, encoder_states)
        return encoder_model


""" Inference: State -> Text """


class TextDecoderGeneratorModelCreator(AbstractModelCreator):

    def __init__(self,
                 decoder_embedding,
                 decoder_lstm,
                 decoder_dense):
        self.decoder_embedding = decoder_embedding
        self.decoder_lstm = decoder_lstm
        self.decoder_dense = decoder_dense

    def create_model(self):
        """ Create text inference decoder model """
        # Each time step will be only single word in the decoder input
        decoder_inputs_single = Input(shape=(1,))
        decoder_inputs_single_x = self.decoder_embedding(decoder_inputs_single)

        decoder_state_input_h = Input(shape=(256,))
        decoder_states_inputs = [decoder_state_input_h]
        decoder_outputs_single, decoder_state_h = self.decoder_lstm(decoder_inputs_single_x,
                                                                    initial_state=decoder_states_inputs)
        decoder_states_outputs = [decoder_state_h]
        decoder_outputs_single = self.decoder_dense(decoder_outputs_single)
        decoder_model = Model(
            [decoder_inputs_single] + decoder_states_inputs,
            [decoder_outputs_single] + decoder_states_outputs)
        return decoder_model
