from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cosine

from generator_models import GeneratorModelCreator
from discriminator_models import DiscriminatorModelCreator
from gan_models import EncoderGanModelCreator
from gan_models import DecoderGanModelCreator

from trainers import EncoderTrainer, DecoderTrainer

from tokenizer import DefaultTokenizer
from displayers import SampleTextDisplayer, SampleImageDisplayer, SampleDiagramDisplayer, SampleReportDisplayer
from bleu_score_calculator import BleuScoreCalculator

from imblearn.over_sampling import RandomOverSampler

from numpy import ones
from numpy import zeros
import numpy as np
import re
import os
import fakesafe_constants as constants

import string
from unicodedata import normalize

import math
import sys

"""
One layer encoding-decoding:
Word -> (Encode) -> Fashion -> (Decode) -> Word
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

print('Load text data')

# Load text data
with open('data/small_vocab_en', 'r', encoding='utf-8') as data_file:
    lines = data_file.read().split('\n')

# Prepare printable data
re_print = re.compile('[^%s]' % re.escape(string.printable))

# Prepare translation table
table = str.maketrans('', '', string.punctuation)

sentences = []
for input_text in lines:
    if input_text:
        # Lower the input text
        input_text = input_text.lower()
        # Remove non-alphabet characters
        input_text = re.sub('[^a-z ]+', '', input_text)
        # Remove extra empty space
        input_text = re.sub(' +', ' ', input_text)
        # Normalizing unicode characters
        input_text = normalize('NFD', input_text).encode('ascii', 'ignore')
        input_text = input_text.decode('UTF-8')
        # Tokenize on white space
        input_text = input_text.split()
        # Removing punctuation
        input_text = [word.translate(table) for word in input_text]
        # Removing non-printable chars form each token
        input_text = [re_print.sub('', w) for w in input_text]
        input_text = " ".join(input_text)
        input_text = input_text.strip()
        sentences.append(input_text)

sentences = np.asarray(sentences)
sentences_train, sentences_test = train_test_split(sentences,
                                                   test_size=0.15)

words_train = []
words_test = []

for sentence in sentences_train:
    words_train += [word for word in sentence.split(' ') if word]
words_train = np.asarray(words_train)

for sentence in sentences_test:
    words_test += [word for word in sentence.split(' ') if word]
words_test = np.asarray(words_test)

oversample = RandomOverSampler(sampling_strategy='minority')
print('Words shape before over sampling: {}'.format(words_train.shape))

words_train = words_train.reshape(-1, 1)
words_train, words_train = oversample.fit_resample(words_train, words_train)
print('Words shape after over sampling: {}'.format(words_train.shape))

# Convert to word embedding
original_embeddings_index = {}
with open('data/glove.6B.50d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        original_embeddings_index[word] = coefs
print('Found %s word vectors.' % len(original_embeddings_index))

embeddings_index = {}
word_embeddings_train = []

for word in words_train:
    embedding_vector = original_embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_index[word] = embedding_vector
        word_embeddings_train.append(embedding_vector)
    else:
        print('Word not found: {}'.format(word))

for word in words_test:
    embedding_vector = original_embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_index[word] = embedding_vector
    else:
        print('Word not found: {}'.format(word))

word_embeddings_train = np.asarray(word_embeddings_train)
print('Word embeddings train: {}'.format(word_embeddings_train.shape))

# Load fashion data
(fashion_image_train, _), _ = fashion_mnist.load_data()
print('Fashion data shape: {}'.format(fashion_image_train.shape))

# Rescale -1 to 1
fashion_image_train_scaled = (fashion_image_train / 255.0) * 2 - 1

"""
Create models
"""

""" Encoder """

# word -> image
word_encoder_creator = GeneratorModelCreator(input_shape=(50,),
                                             output_shape=constants.OUTPUT_SHAPE,
                                             to_image=True,
                                             activation='tanh')
word_encoder = word_encoder_creator.create_model()

# Discriminator
encoder_discriminator_creator = DiscriminatorModelCreator(
    constants.INPUT_SHAPE)
encoder_discriminator = encoder_discriminator_creator.create_model()

# GAN (Combine encoder generator and discriminator)
encoder_gan_creator = EncoderGanModelCreator(word_encoder,
                                             encoder_discriminator)
encoder_gan = encoder_gan_creator.create_model()

""" Decoder """

# image -> word
word_decoder_creator = GeneratorModelCreator(input_shape=constants.INPUT_SHAPE,
                                             output_shape=50,
                                             from_image=True)
word_decoder = word_decoder_creator.create_model()

# GAN (Combine state2image generator and image2state generator)
decoder_gan_creator = DecoderGanModelCreator(word_encoder,
                                             word_decoder)
decoder_gan = decoder_gan_creator.create_model()

"""
Create trainers
"""

""" Encoder """

# word -> image
encoder_trainer = EncoderTrainer(word_encoder,
                                 encoder_discriminator,
                                 encoder_gan,
                                 training_epochs=constants.TRAINING_EPOCHS,
                                 batch_size=constants.BATCH_SIZE)

""" Decoder """

# image -> word
decoder_trainer = DecoderTrainer(word_encoder,
                                 word_decoder,
                                 decoder_gan,
                                 training_epochs=constants.TRAINING_EPOCHS,
                                 batch_size=constants.BATCH_SIZE)

"""
Start training
"""

text_displayer = SampleTextDisplayer()
diagram_displayer = SampleDiagramDisplayer()
report_displayer = SampleReportDisplayer()

encoder_discriminator_loss = []
encoder_discriminator_accuracy = []

encoder_generator_loss = []
encoder_generator_accuracy = []

decoder_loss = []
decoder_accuracy = []

bleu_scores = []
bleu_score_calculator = BleuScoreCalculator(sentences_test)

for current_round in range(constants.TOTAL_TRAINING_ROUND):

    print('************************')
    print('Round: {}'.format(current_round + 1))
    print('************************')

    """ Train """

    # word -> image
    encoder_trainer.train(input_data=word_embeddings_train,
                          exp_output_data=fashion_image_train_scaled)

    # image -> word
    decoder_trainer.train(input_data=word_embeddings_train)

    """ Inference """

    # Select sample of sequences
    sample_indexes = np.random.randint(0,
                                       sentences_test.shape[0],
                                       constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    original_sentences = sentences_test[sample_indexes]

    # Display original corpus
    original_name = '{} - 1 - Original'.format(current_round + 1)
    text_displayer.display_samples(name=original_name,
                                   samples=original_sentences,
                                   should_display_directly=should_display_directly,
                                   should_save_to_file=should_save_to_file)

    decoded_sentences = []

    current_discriminator_loss = 0
    current_discriminator_acc = 0

    current_generator_loss = 0
    current_generator_acc = 0

    current_decoder_loss = 0
    current_decoder_acc = 0

    for sentence_index in range(original_sentences.shape[0]):

        original_sentence = original_sentences[sentence_index]
        original_words = [
            word for word in original_sentence.split(' ') if word]

        original_embeddings = [embeddings_index.get(
            word) for word in original_words]
        original_embeddings = np.asarray(original_embeddings)

        # Select a random batch of images
        image_indexes = np.random.randint(0,
                                          fashion_image_train_scaled.shape[0],
                                          len(original_words))
        sample_images = fashion_image_train_scaled[image_indexes]

        # Encode images
        encoded_sample_images_scaled = word_encoder.predict(
            original_embeddings)

        # Evaluate
        y_zeros = zeros((len(original_words), 1))
        y_ones = ones((len(original_words), 1))
        loss_fake, acc_fake = encoder_discriminator.evaluate(encoded_sample_images_scaled,
                                                             y_zeros)
        loss_real, acc_real = encoder_discriminator.evaluate(sample_images,
                                                             y_ones)

        d_loss, d_acc = 0.5 * \
            np.add(loss_fake, loss_real), 0.5 * np.add(acc_fake, acc_real)
        g_loss, g_acc = encoder_gan.evaluate(original_embeddings, y_ones)

        current_discriminator_loss += d_loss
        current_discriminator_acc += d_acc

        current_generator_loss += g_loss
        current_generator_acc += g_acc

        # Display encoded images
        row = math.ceil(len(original_words) / constants.DISPLAY_COLUMN)
        image_displayer = SampleImageDisplayer(row=row,
                                               column=constants.DISPLAY_COLUMN,
                                               cmap='gray')
        encoded_name = '{} - 2 - Sentence {} - Encoded'.format(
            (current_round + 1), (sentence_index + 1))
        encoded_sample_images = (encoded_sample_images_scaled + 1) / 2 * 255
        encoded_sample_images = encoded_sample_images[:, :, :, 0]
        image_displayer.display_samples(name=encoded_name,
                                        samples=encoded_sample_images,
                                        should_display_directly=should_display_directly,
                                        should_save_to_file=should_save_to_file)

        # Decode images to word
        loss, accuracy = decoder_gan.evaluate(
            original_embeddings, original_embeddings)

        current_decoder_loss += loss
        current_decoder_acc += accuracy

        decoded_embeddings = decoder_gan.predict(original_embeddings)
        decoded_words = []
        for decoded_embedding in decoded_embeddings:
            smallest_similarity = 1
            nearest_word = None
            for word in embeddings_index:
                ref_embedding = embeddings_index.get(word)
                if ref_embedding is not None:
                    # Calculate cosine similarity
                    similarity = cosine(decoded_embedding, ref_embedding)
                    if similarity < smallest_similarity:
                        smallest_similarity = similarity
                        nearest_word = word
            if nearest_word is not None:
                decoded_words.append(nearest_word)

        decoded_sentence = ' '.join(decoded_words)
        decoded_sentences.append(decoded_sentence)

        # Display decoded word
        decoded_name = '{} - 3 - Decoded'.format(current_round + 1)
        text_displayer.display_samples(name=decoded_name,
                                       samples=decoded_sentences,
                                       should_display_directly=should_display_directly,
                                       should_save_to_file=should_save_to_file)

        print('Sample sentence:')
        print(original_sentence)

        print('Decoded sentence:')
        print(decoded_sentence)

    current_discriminator_loss = current_discriminator_loss / \
        (constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    current_discriminator_acc = current_discriminator_acc / \
        (constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    current_generator_loss = current_generator_loss / \
        (constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    current_generator_acc = current_generator_acc / \
        (constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    current_decoder_loss = current_decoder_loss / \
        (constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    current_decoder_acc = current_decoder_acc / \
        (constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)

    encoder_discriminator_loss.append(current_discriminator_loss)
    encoder_discriminator_accuracy.append(current_discriminator_acc)
    encoder_generator_loss.append(current_generator_loss)
    encoder_generator_accuracy.append(current_generator_acc)
    decoder_loss.append(current_decoder_loss)
    decoder_accuracy.append(current_decoder_acc)

    # Calculate BLEU score
    score = bleu_score_calculator.calculate(
        original_sentences, decoded_sentences)
    bleu_scores.append(score)
    print('BLEU score: {}'.format(score))

    report = {
        'encoder_discriminator_loss': current_discriminator_loss,
        'encoder_discriminator_accuracy': current_discriminator_acc,
        'encoder_generator_loss': current_generator_loss,
        'encoder_generator_accuracy': current_generator_acc,
        'decoder_loss': current_decoder_loss,
        'decoder_accuracy': current_decoder_acc,
        'bleu_score': score
    }

    report_name = 'Report - {}'.format(current_round + 1)
    report_displayer.display_samples(name=report_name,
                                     samples=report,
                                     should_display_directly=should_display_directly,
                                     should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Encoder Discriminator Loss',
                                  samples=encoder_discriminator_loss,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Encoder Discriminator Accuracy',
                                  samples=encoder_discriminator_accuracy,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Encoder Generator Loss',
                                  samples=encoder_generator_loss,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Encoder Generator Accuracy',
                                  samples=encoder_generator_accuracy,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Decoder Loss',
                                  samples=decoder_loss,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Decoder Accuracy',
                                  samples=decoder_accuracy,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Bleu Score',
                                  samples=bleu_scores,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

print('Best score: {}'.format(max(bleu_scores)))

word_encoder.save(
    'model/one_layer_sentence_fashion_embedding_encoder_generator.h5')
word_decoder.save(
    'model/one_layer_sentence_fashion_embedding_decoder_generator.h5')
