from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

from models import Text2ImageGeneratorModelCreator, DiscriminatorModelCreator, EncoderGanModelCreator, DecoderGanModelCreator
from trainers import EncoderTrainer, DecoderTrainer

from tokenizer import DefaultTokenizer
from displayers import SampleTextDisplayer, SampleImageDisplayer

import numpy as np
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
for line in lines:
    corpus.append(line)

# Tokenize text data
tokenizer = DefaultTokenizer()
sequences, max_sequence_length, word_index, index_word, vocabulary_size = tokenizer.tokenize_corpus(
    corpus)

sequences_train, sequences_test = train_test_split(sequences,
                                                   test_size=0.15)
print('Sequence train shape: {}'.format(sequences_train.shape))
print('Sequence test shape: {}'.format(sequences_test.shape))

# Load fashion data
(fashion_image_train, _), _ = fashion_mnist.load_data()
print('Fashion data shape: {}'.format(fashion_image_train.shape))

# Rescale -1 to 1
fashion_image_train_scaled = (fashion_image_train / 255.0) * 2 - 1

"""
Create models
"""

# Encoder

# Create encoder generator
encoder_generator_creator = Text2ImageGeneratorModelCreator(vocabulary_size,
                                                            max_sequence_length,
                                                            constants.OUTPUT_SHAPE)
encoder_generator = encoder_generator_creator.create_model()

# Create encoder discriminator
encoder_discriminator_creator = DiscriminatorModelCreator(
    constants.INPUT_SHAPE)
encoder_discriminator = encoder_discriminator_creator.create_model()

# Create GAN model to combine encoder generator and discriminator
encoder_gan_creator = EncoderGanModelCreator(encoder_generator,
                                             encoder_discriminator)
encoder_gan = encoder_gan_creator.create_model()

# Decoder

# # Create decoder generator
# decoder_generator_creator = Image2ImageGeneratorModelCreator(
#     constants.INPUT_SHAPE)
# decoder_generator = decoder_generator_creator.create_model()

# # Create GAN model to combine encoder generator and decoder generator
# decoder_gan_creator = DecoderGanModelCreator(encoder_generator,
#                                              decoder_generator)
# decoder_gan = decoder_gan_creator.create_model()

"""
Create trainers
"""

# Encoder

# Create encoder trainer
encoder_trainer = EncoderTrainer(encoder_generator,
                                 encoder_discriminator,
                                 encoder_gan,
                                 training_epochs=constants.TRAINING_EPOCHS,
                                 batch_size=constants.BATCH_SIZE,
                                 input_data=sequences_train,
                                 exp_output_data=fashion_image_train_scaled)

# Decoder

# Create decoder trainer
# decoder_trainer = DecoderTrainer(encoder_generator,
#                                  decoder_generator,
#                                  decoder_gan,
#                                  training_epochs=constants.TRAINING_EPOCHS,
#                                  batch_size=constants.BATCH_SIZE,
#                                  input_data=fashion_image_train_scaled)

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

    # Train encoder
    encoder_trainer.train_model()
    # # Train decoder
    # decoder_trainer.train_model()

    # Select sample of sequences
    sample_indexes = np.random.randint(0,
                                       sequences_test.shape[0],
                                       constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    sample_sequences = sequences_test[sample_indexes]

    sample_corpus = []
    for sequence in sample_sequences:
        words = []
        for index in sequence:
            word = index_word.get(index)
            if word:
                words.append(word)
            else:
                # Since we have post padding index "0" at the end of each sequence
                # If there is post padding index "0", it means we have reached the end of each sequence
                # Then we may simply skip the current iteration, and start the next sentence
                break
        sample_corpus.append(' '.join(words))

    # Display original corpus
    original_name = '{} - 1 - Original'.format(current_round + 1)
    text_displayer.display_samples(name=original_name,
                                   samples=sample_corpus,
                                   should_display_directly=should_display_directly,
                                   should_save_to_file=should_save_to_file)

    # Encode images
    encoded_sample_images_scaled = encoder_generator.predict(sample_sequences)

    # Display encoded images
    encoded_name = '{} - 2 - Encoded'.format(current_round + 1)
    encoded_sample_images = (encoded_sample_images_scaled + 1) / 2 * 255
    encoded_sample_images = encoded_sample_images[:, :, :, 0]
    image_displayer.display_samples(name=encoded_name,
                                    samples=encoded_sample_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)

    # # Decode images
    # decoded_sample_images_scaled = decoder_gan.predict(sample_images_scaled)

    # # Display decoded images
    # decoded_name = 'Decoded - {}'.format(current_round + 1)
    # decoded_sample_images = (decoded_sample_images_scaled + 1) / 2 * 255
    # decoded_sample_images = decoded_sample_images[:, :, :, 0]
    # image_displayer.display_samples(name=decoded_name,
    #                                 samples=decoded_sample_images,
    #                                 should_display_directly=should_display_directly,
    #                                 should_save_to_file=should_save_to_file)
