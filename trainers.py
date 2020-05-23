from numpy import ones
from numpy import zeros
import numpy as np

from abc import ABC, abstractmethod
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

"""
Abstract Model Trainer
"""


class AbstractModelTrainer(ABC):

    @abstractmethod
    # TODO: Deprecated
    def train_model(self):
        raise NotImplementedError('Abstract class shall not be implemented')

    @abstractmethod
    def train(self, input_data, exp_output_data=None):
        raise NotImplementedError('Abstract class shall not be implemented')


"""
Encoder Trainer
"""


class EncoderTrainer(AbstractModelTrainer):

    def __init__(self,
                 encoder_generator,
                 encoder_discriminator,
                 encoder_gan,
                 training_epochs,
                 batch_size,
                 input_data,  # Input data of encoder
                 exp_output_data):  # Expected output data of encoder

        self.encoder_generator = encoder_generator
        self.encoder_discriminator = encoder_discriminator
        self.encoder_gan = encoder_gan

        self.training_epochs = training_epochs
        self.batch_size = batch_size

        self.input_data = input_data
        self.exp_output_data = exp_output_data

        self.y_zeros = zeros((self.batch_size, 1))
        self.y_ones = ones((self.batch_size, 1))

    def train(self, input_data, exp_output_data):

        print('========== Start encoder training ==========')
        for current_epoch in range(self.training_epochs):

            # Select a random batch of data
            input_indexes = np.random.randint(0,
                                              input_data.shape[0],
                                              self.batch_size)
            x_input = input_data[input_indexes]

            output_indexes = np.random.randint(0,
                                               exp_output_data.shape[0],
                                               self.batch_size)
            x_exp_output = exp_output_data[output_indexes]

            # Generate output data via encoder generator
            x_gen_output = self.encoder_generator.predict(x_input)

            # ---------------------
            #  Train encoder discriminator
            # ---------------------
            d_loss, d_acc = self.__train_encoder_discriminator(x_gen_output,
                                                               x_exp_output)

            # ---------------------
            #  Train encoder generator
            # ---------------------
            g_loss, g_acc = self.__train_encoder_generator(x_input)

            # Plot the progress
            print('[Encoder] - epochs: {}, d_loss: {}, d_acc: {}, g_loss: {}, g_acc: {}'.format(
                (current_epoch + 1), d_loss, d_acc, g_loss, g_acc))

    # TODO: Deprecated
    def train_model(self):
        self.train(self.input_data, self.exp_output_data)

    def __train_encoder_discriminator(self, x_gen_output, x_exp_output):

        # Generated output is marked as 0
        loss_fake, acc_fake = self.encoder_discriminator.train_on_batch(x_gen_output,
                                                                        self.y_zeros)

        # Expected output is marked as 1
        loss_real, acc_real = self.encoder_discriminator.train_on_batch(x_exp_output,
                                                                        self.y_ones)

        return 0.5 * np.add(loss_fake, loss_real), 0.5 * np.add(acc_fake, acc_real)

    def __train_encoder_generator(self, x_input):

        # Train generator via GAN model
        loss, accuracy = self.encoder_gan.train_on_batch(x_input, self.y_ones)
        return loss, accuracy


class TextEncoderTrainer(AbstractModelTrainer):

    def __init__(self,
                 text_encoder_generator,
                 state_encoder_generator,
                 encoder_discriminator,
                 encoder_gan,
                 training_epochs,
                 batch_size,
                 input_data,  # Input data of encoder
                 exp_output_data):  # Expected output data of encoder

        self.text_encoder_generator = text_encoder_generator
        self.state_encoder_generator = state_encoder_generator
        self.encoder_discriminator = encoder_discriminator
        self.encoder_gan = encoder_gan

        self.training_epochs = training_epochs
        self.batch_size = batch_size

        self.input_data = input_data
        self.exp_output_data = exp_output_data

        self.y_zeros = zeros((self.batch_size, 1))
        self.y_ones = ones((self.batch_size, 1))

    def train(self, input_data, exp_output_data):

        print('========== Start text encoder training ==========')
        for current_epoch in range(self.training_epochs):

            # Select a random batch of data
            input_indexes = np.random.randint(0,
                                              input_data.shape[0],
                                              self.batch_size)
            x_input = input_data[input_indexes]

            output_indexes = np.random.randint(0,
                                               exp_output_data.shape[0],
                                               self.batch_size)
            x_exp_output = exp_output_data[output_indexes]

            # Generate output data via encoder generator
            x_state_output = self.text_encoder_generator.predict(x_input)
            x_gen_output = self.state_encoder_generator.predict(x_state_output)

            # ---------------------
            #  Train encoder discriminator
            # ---------------------
            d_loss, d_acc = self.__train_encoder_discriminator(x_gen_output,
                                                               x_exp_output)

            # ---------------------
            #  Train encoder generator
            # ---------------------
            g_loss, g_acc = self.__train_encoder_generator(x_input)

            # Plot the progress
            print('[Encoder] - epochs: {}, d_loss: {}, d_acc: {}, g_loss: {}, g_acc: {}'.format(
                (current_epoch + 1), d_loss, d_acc, g_loss, g_acc))

    # TODO: Deprecated
    def train_model(self):
        self.train(self.input_data, self.exp_output_data)

    def __train_encoder_discriminator(self, x_gen_output, x_exp_output):

        # Generated output is marked as 0
        loss_fake, acc_fake = self.encoder_discriminator.train_on_batch(x_gen_output,
                                                                        self.y_zeros)

        # Expected output is marked as 1
        loss_real, acc_real = self.encoder_discriminator.train_on_batch(x_exp_output,
                                                                        self.y_ones)

        return 0.5 * np.add(loss_fake, loss_real), 0.5 * np.add(acc_fake, acc_real)

    def __train_encoder_generator(self, x_input):

        # Train generator via GAN model
        loss, accuracy = self.encoder_gan.train_on_batch(x_input, self.y_ones)
        return loss, accuracy


"""
Decoder Trainer
"""


class DecoderTrainer(AbstractModelTrainer):

    def __init__(self,
                 encoder_generator,
                 decoder_generator,
                 decoder_gan,
                 training_epochs,
                 batch_size,
                 input_data):  # Input data of decoder

        self.encoder_generator = encoder_generator
        self.decoder_generator = decoder_generator
        self.decoder_gan = decoder_gan

        self.training_epochs = training_epochs
        self.batch_size = batch_size

        self.input_data = input_data

    def train(self, input_data, exp_output_data):

        print('========== Start decoder training ==========')
        for current_epoch in range(self.training_epochs):

            # Select a random batch of data
            input_indexes = np.random.randint(0,
                                              input_data.shape[0],
                                              self.batch_size)
            x_input = input_data[input_indexes]

            # ---------------------
            #  Train decoder generator
            # ---------------------
            loss, accuracy = self.__train_decoder_generator(x_input)

            # Plot the progress
            print('[Decoder] - epochs: {}, loss: {}, accuracy: {}'.format(
                (current_epoch + 1), loss, accuracy))

    # TODO: Deprecated
    def train_model(self):
        self.train(self.input_data, None)

    def __train_decoder_generator(self, x_input):

        # Train decoder generator via GAN model
        # Decoder GAN model has 2 layers: (1) encoder generator -> (2) decoder generator
        # The input data will be first encoded via encoder generator
        # Then, it will be further decoded via decoder generator
        # We should train the decoder generator to output data same as input data
        x_output = x_input
        loss, accuracy = self.decoder_gan.train_on_batch(x_input, x_output)
        return loss, accuracy


class TextDecoderTrainer(AbstractModelTrainer):

    def __init__(self,
                 encoder_generator,
                 decoder_generator,
                 training_epochs,
                 batch_size,
                 input_data,  # Input data of decoder
                 max_sequence_length,
                 vocabulary_size):

        self.encoder_generator = encoder_generator
        self.decoder_generator = decoder_generator

        self.training_epochs = training_epochs
        self.batch_size = batch_size

        self.input_data = input_data
        self.max_sequence_length = max_sequence_length
        self.vocabulary_size = vocabulary_size

    def train(self, input_data, exp_output_data):

        print('========== Start text decoder training ==========')
        for current_epoch in range(self.training_epochs):

            # Select a random batch of data
            input_indexes = np.random.randint(0,
                                              input_data.shape[0],
                                              self.batch_size)
            sequences = input_data[input_indexes]

            # Create data generator
            x_sequences, y_targets = self.__create_data_generator(sequences)

            # ---------------------
            #  Get hidden states of each sequence
            # ---------------------
            hidden_states = self.encoder_generator.predict(x_sequences)

            # ---------------------
            #  Train decoder generator
            # ---------------------
            loss, accuracy = self.decoder_generator.train_on_batch([hidden_states, x_sequences],
                                                                   y_targets)

            print('[Decoder] - epochs: {}, loss: {}, accuracy: {}'.format(
                (current_epoch + 1), loss, accuracy))

    def train_model(self):
        self.train(self.input_data, None)

    def __create_data_generator(self, sequences):

        input_sequences = []
        output_targets = []

        for i in range(self.batch_size):

            sequence = sequences[i]
            for j in range(1, self.max_sequence_length):

                # Split the sequence into input and output pair
                # For example:
                # For the sentence "I am handsome"
                # If the input is "I", then the output target should be "am"
                # If the input is "I am", then the output target should be "handsome"
                input_sequence = sequence[:j]
                output_target = sequence[j]

                if output_target == 0:
                    # Since the we use "0" to pad the sequences to the same length
                    # For example, if the sentence has 3 words while the longest one has 6 words,
                    # then the sequences will be like [1, 2, 3, 0, 0, 0]
                    # Therefore, if the target word index is "0",
                    # Then it means there will be no more words in the sentence
                    break

                # Create input and output data
                input_sequences.append(input_sequence)
                output_targets.append(output_target)

        # Pad sequence
        input_sequences = pad_sequences(input_sequences,
                                        maxlen=self.max_sequence_length,
                                        padding='post')

        output_targets = to_categorical(
            output_targets, num_classes=self.vocabulary_size)
        x_sequences = np.asarray(input_sequences)
        y_targets = np.asarray(output_targets)

        return x_sequences, y_targets
