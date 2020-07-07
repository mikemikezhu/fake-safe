from numpy import ones
from numpy import zeros
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from abc import ABC, abstractmethod

"""
Abstract Model Trainer
"""


class AbstractModelTrainer(ABC):

    @abstractmethod
    def train(self,
              input_data,
              exp_output_data=None,
              eval_input_data=None,
              eval_output_data=None):
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
                 batch_size):

        self.encoder_generator = encoder_generator
        self.encoder_discriminator = encoder_discriminator
        self.encoder_gan = encoder_gan

        self.training_epochs = training_epochs
        self.batch_size = batch_size

        self.y_zeros = zeros((self.batch_size, 1))
        self.y_ones = ones((self.batch_size, 1))

    def train(self,
              input_data,
              exp_output_data,
              eval_input_data=None,
              eval_output_data=None):

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
                 batch_size):

        self.encoder_generator = encoder_generator
        self.decoder_generator = decoder_generator
        self.decoder_gan = decoder_gan

        self.training_epochs = training_epochs
        self.batch_size = batch_size

    def train(self,
              input_data,
              exp_output_data=None,
              eval_input_data=None,
              eval_output_data=None):

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

    def __train_decoder_generator(self, x_input):

        # Train decoder generator via GAN model
        # Decoder GAN model has 2 layers: (1) encoder generator -> (2) decoder generator
        # The input data will be first encoded via encoder generator
        # Then, it will be further decoded via decoder generator
        # We should train the decoder generator to output data same as input data
        x_output = x_input
        loss, accuracy = self.decoder_gan.train_on_batch(x_input, x_output)
        return loss, accuracy


"""
Seq2Seq Trainer
"""


class Seq2SeqTrainer(AbstractModelTrainer):

    def __init__(self,
                 seq2seq_model,
                 training_epochs,
                 batch_size):

        self.seq2seq_model = seq2seq_model
        self.training_epochs = training_epochs
        self.batch_size = batch_size

    def train(self,
              input_data,  # [input_encoder, input_decoder]
              exp_output_data,  # decoder target predictions
              eval_input_data=None,
              eval_output_data=None):

        print('========== Start seq2seq training ==========')

        input_data_encoder, input_data_decoder = input_data
        input_length = input_data_encoder.shape[0]

        for current_epoch in range(self.training_epochs):

            # Select a random batch of data
            input_indexes = np.random.randint(0,
                                              input_length,
                                              self.batch_size)

            x_input_encoder = input_data_encoder[input_indexes]
            x_input_decoder = input_data_decoder[input_indexes]

            x_input = [x_input_encoder, x_input_decoder]
            y_input = exp_output_data[input_indexes]

            # ---------------------
            #  Train seq2seq model
            # ---------------------
            loss, accuracy = self.seq2seq_model.train_on_batch(x_input,
                                                               y_input)

            # Plot the progress
            print('[Seq2Seq] - epochs: {}, loss: {}, accuracy: {}'.format(
                (current_epoch + 1), loss, accuracy))


"""
Classifier Trainer
"""


class ClassifierTrainer(AbstractModelTrainer):

    def __init__(self,
                 classifier,
                 training_epochs,
                 batch_size,
                 should_early_stopping=True,
                 should_reduce_learning_rate=True):
        self.classifier = classifier
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.should_early_stopping = should_early_stopping
        self.should_reduce_learning_rate = should_reduce_learning_rate

    def train(self,
              input_data,  # x_train
              exp_output_data,  # y_train
              eval_input_data,  # x_test
              eval_output_data):  # y_test

        print('========== Start classifier training ==========')
        # Callbacks
        callbacks = None
        if self.should_early_stopping or self.should_reduce_learning_rate:
            callbacks = []
            if self.should_early_stopping:
                early_stopping = EarlyStopping(monitor='val_accuracy',
                                               patience=10)
                callbacks.append(early_stopping)

            if self.should_reduce_learning_rate:
                reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                                         factor=0.9,
                                                         patience=5)
                callbacks.append(reduce_learning_rate)

        # Start training
        history = self.classifier.fit(input_data,
                                      exp_output_data,
                                      epochs=self.training_epochs,
                                      batch_size=self.batch_size,
                                      validation_data=(eval_input_data,
                                                       eval_output_data),
                                      callbacks=callbacks)
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        return accuracy, val_accuracy
