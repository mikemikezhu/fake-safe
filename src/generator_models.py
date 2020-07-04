from abstract_models import AbstractModelCreator
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, LeakyReLU, Reshape
from tensorflow.keras.layers import Input, Embedding, GRU

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


""" Word / Text State -> Image """


class DefaultEncoderGeneratorModelCreator(AbstractModelCreator):

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def create_model(self):

        model = Sequential()

        model.add(Dense(64, input_shape=self.input_shape))
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

        model.add(Dense(np.prod(self.output_shape),
                        activation='tanh'))
        model.add(Reshape(self.output_shape))

        print('Word to Image Generator model:')
        model.summary()

        return model


""" Image -> Word / Text State """


class DefaultDecoderGeneratorModelCreator(AbstractModelCreator):

    def __init__(self, input_shape, output_shape, activation='linear'):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation

    def create_model(self):

        # Image model to extract image features
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

        model.add(Dense(self.output_shape,
                        activation=self.activation))

        print('Image to Word Generator model:')
        model.summary()

        return model


""" Sentence -> State -> Sentence """


class Seq2SeqModelCreator(AbstractModelCreator):

    def __init__(self, vocabulary_size, max_sequence_length):
        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = max_sequence_length

    def create_model(self):

        # Create train and inference model
        """
        Train
        """

        """ 1) Train - Encoder """

        # Define an input sequence and process it
        encoder_inputs = Input(shape=(self.max_sequence_length,),
                               name='encoder_inputs')
        encoder_embedding = Embedding(input_dim=self.vocabulary_size,
                                      output_dim=200,
                                      mask_zero=True,
                                      name='encoder_embedding')(encoder_inputs)
        encoder = GRU(256,
                      activation='relu',
                      return_sequences=True,
                      return_state=True,
                      name='encoder_GRU')
        _, encoder_states = encoder(encoder_embedding)

        """ 2) Train - Decoder """

        # Set up the decoder, using `encoder_states` as initial state
        decoder_inputs = Input(shape=((self.max_sequence_length - 1),),
                               name='decoder_inputs')
        decoder_embedding = Embedding(input_dim=self.vocabulary_size,
                                      output_dim=200,
                                      mask_zero=True,
                                      name='decoder_embedding')(decoder_inputs)
        decoder_gru = GRU(256,
                          activation='relu',
                          return_sequences=True,
                          return_state=True,
                          name='decoder_GRU')

        decoder_outputs, _ = decoder_gru(decoder_embedding,
                                         initial_state=encoder_states)
        decoder_dense = Dense(self.vocabulary_size,
                              activation='softmax',
                              name='decoder_dense')
        decoder_predictions = decoder_dense(decoder_outputs)
        print(decoder_predictions.shape)

        # Define the model
        train_model = Model([encoder_inputs, decoder_inputs],
                            decoder_predictions,
                            name='train_model')

        print('[Train] Sentence to State to Sentence Generator model:')
        train_model.summary()

        # Compile the train model
        train_model.compile(loss='sparse_categorical_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])

        """
        Inference
        """

        """ 1) Inference - Encoder """

        # Create inference encoder model
        encoder_model = Model(encoder_inputs,
                              encoder_states,
                              name='encoder_model')

        print('[Inference - Encoder] Sentence to State Generator model:')
        encoder_model.summary()

        """ 2) Inference - Decoder """

        # Create inference decoder model
        input_states = Input(shape=(256,),
                             name='inference_states')
        outputs, output_states = decoder_gru(decoder_embedding,
                                             initial_state=input_states)
        predictions = decoder_dense(outputs)

        decoder_model = Model([decoder_inputs] + [input_states],
                              [predictions] + [output_states],
                              name='decoder_model')

        print('[Inference - Decoder] State to Sentence Generator model:')
        decoder_model.summary()

        return train_model, encoder_model, decoder_model
