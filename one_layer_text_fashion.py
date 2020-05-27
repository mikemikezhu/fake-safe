from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from generator_models import TextEncoderGeneratorModelCreator
from generator_models import TextDecoderGeneratorModelCreator
from discriminator_models import DiscriminatorModelCreator
from gan_models import EncoderGanModelCreator
from gan_models import DecoderGanModelCreator

from trainers import EncoderTrainer, DecoderTrainer

from tokenizer import DefaultTokenizer
from displayers import SampleTextDisplayer, SampleImageDisplayer

import numpy as np
import re
import constants

import sys

"""
One layer encoding-decoding:
Text -> (Encode) -> Fashion -> (Decode) -> Text
"""

"""
Parse arguments
"""
try:
    should_display_directly = int(sys.argv[1])
    should_save_to_file = int(sys.argv[2])
except ValueError:
    print('Invalid system argument')
    should_display_directly = False
    should_save_to_file = False

should_display_directly = False if should_display_directly == 0 else True
print('Should display directly: {}'.format(should_display_directly))

should_save_to_file = False if should_save_to_file == 0 else True
print('Should save to file: {}'.format(should_save_to_file))

"""
Load data
"""

# Load text data
with open(constants.TEXT_2_IMAGE_DATASET_PATH, 'r') as data_file:
    lines = data_file.read().split('\n')

corpus = []
words = []
for line in lines:
    line = line.strip()  # Remove "\n" and empty space
    line = line.lower()  # Lower all characters
    line = re.sub('[^a-z ]+', '', line)  # Remove non-alphabet characters
    line = re.sub(' +', ' ', line)  # Remove extra empty space
    words += [word for word in line.split(' ') if word]
    corpus.append(line)

# Tokenize text data
tokenizer = DefaultTokenizer()
sequences, max_sequence_length, word_index, index_word, vocabulary_size = tokenizer.tokenize_corpus(
    corpus)

tokens = [word_index[word] for word in words]
tokens = np.asarray(tokens)
tokens_train, tokens_test = train_test_split(tokens,
                                             test_size=0.15)
print('Tokens train shape: {}'.format(tokens_train.shape))
print('Tokens test shape: {}'.format(tokens_test.shape))

# Load fashion data
(fashion_image_train, _), _ = fashion_mnist.load_data()
print('Fashion data shape: {}'.format(fashion_image_train.shape))

# Rescale -1 to 1
fashion_image_train_scaled = (fashion_image_train / 255.0) * 2 - 1

"""
Create models
"""

""" Encoder """

# text -> image
text_encoder_creator = TextEncoderGeneratorModelCreator(constants.INPUT_SHAPE,
                                                        vocabulary_size)
text_encoder = text_encoder_creator.create_model()

# Discriminator
encoder_discriminator_creator = DiscriminatorModelCreator(
    constants.INPUT_SHAPE)
encoder_discriminator = encoder_discriminator_creator.create_model()

# GAN (Combine encoder generator and discriminator)
encoder_gan_creator = EncoderGanModelCreator(text_encoder,
                                             encoder_discriminator)
encoder_gan = encoder_gan_creator.create_model()

""" Decoder """

# image -> text
text_decoder_creator = TextDecoderGeneratorModelCreator(constants.INPUT_SHAPE,
                                                        vocabulary_size)
text_decoder = text_decoder_creator.create_model()

# GAN (Combine state2image generator and image2state generator)
decoder_gan_creator = DecoderGanModelCreator(text_encoder,
                                             text_decoder,
                                             loss='sparse_categorical_crossentropy')
decoder_gan = decoder_gan_creator.create_model()

"""
Create trainers
"""

""" Encoder """

# text -> image
encoder_trainer = EncoderTrainer(text_encoder,
                                 encoder_discriminator,
                                 encoder_gan,
                                 training_epochs=constants.TRAINING_EPOCHS,
                                 batch_size=constants.BATCH_SIZE,
                                 input_data=tokens_train,
                                 exp_output_data=fashion_image_train_scaled)

""" Decoder """

# image -> text
decoder_trainer = DecoderTrainer(text_encoder,
                                 text_decoder,
                                 decoder_gan,
                                 training_epochs=constants.TRAINING_EPOCHS,
                                 batch_size=constants.BATCH_SIZE,
                                 input_data=tokens_train)

"""
Start training
"""

text_displayer = SampleTextDisplayer()
image_displayer = SampleImageDisplayer(row=constants.DISPLAY_ROW,
                                       column=constants.DISPLAY_COLUMN)

for current_round in range(constants.TOTAL_TRAINING_ROUND):

    print('************************')
    print('Round: {}'.format(current_round + 1))
    print('************************')

    """ Train """

    # text -> image
    encoder_trainer.train_model()

    # image -> text
    decoder_trainer.train_model()

    """ Inference """

    # Select sample of sequences
    sample_indexes = np.random.randint(0,
                                       tokens_test.shape[0],
                                       constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    sample_tokens = tokens_test[sample_indexes]

    sample_words = []
    for token in sample_tokens:
        word = index_word.get(token)
        if word:
            sample_words.append(word)

    # Display original corpus
    original_name = '{} - 1 - Original'.format(current_round + 1)
    text_displayer.display_samples(name=original_name,
                                   samples=sample_words,
                                   should_display_directly=should_display_directly,
                                   should_save_to_file=should_save_to_file)

    # Encode images
    encoded_sample_images_scaled = text_encoder.predict(sample_tokens)

    # Display encoded images
    encoded_name = '{} - 2 - Encoded'.format(current_round + 1)
    encoded_sample_images = (encoded_sample_images_scaled + 1) / 2 * 255
    encoded_sample_images = encoded_sample_images[:, :, :, 0]
    image_displayer.display_samples(name=encoded_name,
                                    samples=encoded_sample_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)

    # Decode images to text
    loss, accuracy = decoder_gan.evaluate(sample_tokens, sample_tokens)
    print('Decode loss: {}, accuracy: {}'.format(loss, accuracy))
    tokens_probs = decoder_gan.predict(sample_tokens)
    decoded_words = []
    for probs in tokens_probs:
        token = np.argmax(probs)
        word = index_word[token]
        decoded_words.append(word)

    print(sample_words)
    print(decoded_words)

    # Display decoded text
    decoded_name = '{} - 3 - Decoded'.format(current_round + 1)
    text_displayer.display_samples(name=decoded_name,
                                   samples=decoded_words,
                                   should_display_directly=should_display_directly,
                                   should_save_to_file=should_save_to_file)
