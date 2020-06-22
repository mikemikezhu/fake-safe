from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

from generator_models import Seq2SeqModelCreator
from generator_models import DefaultEncoderGeneratorModelCreator
from generator_models import DefaultDecoderGeneratorModelCreator
from discriminator_models import DiscriminatorModelCreator
from gan_models import EncoderGanModelCreator
from gan_models import DecoderGanModelCreator

from trainers import Seq2SeqTrainer, EncoderTrainer, DecoderTrainer
from displayers import SampleTextDisplayer, SampleImageDisplayer, SampleDiagramDisplayer
from bleu_score_calculator import BleuScoreCalculator

from tokenizer import DefaultTokenizer

from numpy import ones
from numpy import zeros
import numpy as np
import re
import constants

import string
from unicodedata import normalize

import sys

"""
One layer encoding-decoding:
Sentence -> (Encode) -> Fashion -> (Decode) -> Sentence
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
with open(constants.TEXT_2_IMAGE_DATASET_PATH, 'r', encoding='utf-8') as data_file:
    lines = data_file.read().split('\n')

# Prepare printable data
re_print = re.compile('[^%s]' % re.escape(string.printable))

# Prepare translation table
table = str.maketrans('', '', string.punctuation)

sentences = []
for input_text in lines[:80000]:
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
        # Encoder start and end
        input_text = 'startofsentence ' + input_text + ' endofsentence'
        sentences.append(input_text)

print('Tokenize text data')

# Tokenize text data
tokenizer = DefaultTokenizer()
sentences_tokenized, max_sequence_length, word_index, index_word, vocabulary_size = tokenizer.tokenize_corpus(
    sentences)

sentences_tokenized_reshape = sentences_tokenized.reshape(
    (-1, max_sequence_length, 1))
print(sentences_tokenized_reshape.shape)

sentences_input = sentences_tokenized_reshape[:, :-1, :]
sentences_target = sentences_tokenized_reshape[:, 1:, :]

print(sentences_tokenized.shape)
print(sentences_input.shape)
print(sentences_target.shape)

print('Prepare train and test data')

input_encoder_train, input_encoder_test = train_test_split(sentences_tokenized,
                                                           test_size=0.15,
                                                           random_state=1)

input_decoder_train, input_decoder_test = train_test_split(sentences_input,
                                                           test_size=0.15,
                                                           random_state=1)

target_decoder_train, target_decoder_test = train_test_split(sentences_target,
                                                             test_size=0.15,
                                                             random_state=1)

x_train = [input_encoder_train, input_decoder_train]
x_test = [input_encoder_test, input_decoder_test]

y_train = target_decoder_train
y_test = target_decoder_test

print('Tokens input encoder train shape: {}'.format(input_encoder_train.shape))
print('Tokens input decoder train shape: {}'.format(input_decoder_train.shape))

print('Tokens input encoder test shape: {}'.format(input_encoder_test.shape))
print('Tokens input decoder test shape: {}'.format(input_decoder_test.shape))

print('Tokens target decoder train shape: {}'.format(target_decoder_train.shape))
print('Tokens target decoder test shape: {}'.format(target_decoder_test.shape))

# Load fashion data
(fashion_image_train, _), _ = fashion_mnist.load_data()
print('Fashion data shape: {}'.format(fashion_image_train.shape))

# Rescale -1 to 1
fashion_image_train_scaled = (fashion_image_train / 255.0) * 2 - 1

"""
Create models
"""

""" Seq2Seq """

# Sentence -> State -> Sentence

seq2seq_model_creator = Seq2SeqModelCreator(vocabulary_size,
                                            max_sequence_length)
seq2seq_train_model, seq2seq_encoder_model, seq2seq_decoder_model = seq2seq_model_creator.create_model()

""" Encoder """

# state -> image
state_encoder_creator = DefaultEncoderGeneratorModelCreator(input_shape=(256,),
                                                            output_shape=constants.OUTPUT_SHAPE)
state_encoder = state_encoder_creator.create_model()

# Discriminator
encoder_discriminator_creator = DiscriminatorModelCreator(
    constants.INPUT_SHAPE)
encoder_discriminator = encoder_discriminator_creator.create_model()

# GAN (Combine encoder generator and discriminator)
encoder_gan_creator = EncoderGanModelCreator(state_encoder,
                                             encoder_discriminator)
encoder_gan = encoder_gan_creator.create_model()

""" Decoder """

# image -> state
state_decoder_creator = DefaultDecoderGeneratorModelCreator(input_shape=constants.INPUT_SHAPE,
                                                            output_shape=256,
                                                            activation='linear')
state_decoder = state_decoder_creator.create_model()

# GAN (Combine state2image generator and image2state generator)
decoder_gan_creator = DecoderGanModelCreator(state_encoder,
                                             state_decoder)
decoder_gan = decoder_gan_creator.create_model()

"""
Create trainers
"""

""" Seq2Seq """

# Sentence -> State -> Sentence

seq2seq_trainer = Seq2SeqTrainer(seq2seq_train_model,
                                 training_epochs=constants.TRAINING_EPOCHS,
                                 batch_size=constants.BATCH_SIZE)

""" Encoder """

# word -> image
encoder_trainer = EncoderTrainer(state_encoder,
                                 encoder_discriminator,
                                 encoder_gan,
                                 training_epochs=constants.TRAINING_EPOCHS,
                                 batch_size=constants.BATCH_SIZE)

""" Decoder """

# image -> word
decoder_trainer = DecoderTrainer(state_encoder,
                                 state_decoder,
                                 decoder_gan,
                                 training_epochs=constants.TRAINING_EPOCHS,
                                 batch_size=constants.BATCH_SIZE)

"""
Start training
"""

text_displayer = SampleTextDisplayer()
diagram_displayer = SampleDiagramDisplayer()
image_displayer = SampleImageDisplayer(row=constants.DISPLAY_ROW,
                                       column=constants.DISPLAY_COLUMN,
                                       cmap='gray')

seq2seq_loss = []
seq2seq_accuracy = []

encoder_discriminator_loss = []
encoder_discriminator_accuracy = []

encoder_generator_loss = []
encoder_generator_accuracy = []

decoder_loss = []
decoder_accuracy = []

bleu_scores = []

y_zeros = zeros((constants.DISPLAY_ROW * constants.DISPLAY_COLUMN, 1))
y_ones = ones((constants.DISPLAY_ROW * constants.DISPLAY_COLUMN, 1))

original_sentences = []
for sentence in sentences:
    words = [word for word in sentence.split(' ') if word !=
             'startofsentence' and word != 'endofsentence']
    original_sentence = ' '.join(words)
    original_sentences.append(original_sentence)
bleu_score_calculator = BleuScoreCalculator(original_sentences)

for current_round in range(constants.TOTAL_TRAINING_ROUND):

    print('************************')
    print('Round: {}'.format(current_round + 1))
    print('************************')

    """ Train """

    # sentence -> state -> sentence
    seq2seq_trainer.train(input_data=x_train,
                          exp_output_data=y_train)

    # state -> image
    states = seq2seq_encoder_model.predict(x_train)
    encoder_trainer.train(input_data=states,
                          exp_output_data=fashion_image_train_scaled)

    # image -> state
    decoder_trainer.train(input_data=states)

    """ Inference """

    ####################
    # Select samples
    ####################

    # Select a random batch of sentences
    sample_indexes = np.random.randint(0,
                                       input_encoder_test.shape[0],
                                       constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)

    sample_input_encoder = input_encoder_test[sample_indexes]
    sample_input_decoder = input_decoder_test[sample_indexes]

    sample_input = [sample_input_encoder, sample_input_decoder]
    sample_target = target_decoder_test[sample_indexes]

    # Select a random batch of images
    image_indexes = np.random.randint(0,
                                      fashion_image_train_scaled.shape[0],
                                      constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    sample_images = fashion_image_train_scaled[image_indexes]

    ####################
    # Seq2Seq
    ####################

    sample_sentences = []
    for sample_sequence in sample_input_encoder:
        sample_words = [index_word[token]
                        for token in sample_sequence if token != 0]
        sample_words = [word for word in sample_words if word !=
                        'startofsentence' and word != 'endofsentence']
        sample_sentence = ' '.join(sample_words)
        sample_sentences.append(sample_sentence)

    # Display original corpus
    original_name = '{} - 1 - Original'.format(current_round + 1)
    text_displayer.display_samples(name=original_name,
                                   samples=sample_sentences,
                                   should_display_directly=should_display_directly,
                                   should_save_to_file=should_save_to_file)

    # Evaluate
    loss_seq2seq, acc_seq2seq = seq2seq_train_model.evaluate(sample_input,
                                                             sample_target)
    seq2seq_loss.append(loss_seq2seq)
    seq2seq_accuracy.append(acc_seq2seq)

    ####################
    # Encode image
    ####################

    # sentence -> state
    sample_states = seq2seq_encoder_model.predict(sample_input_encoder)

    # state -> image
    encoded_sample_images_scaled = state_encoder.predict(sample_states)

    # Evaluate
    loss_fake, acc_fake = encoder_discriminator.evaluate(encoded_sample_images_scaled,
                                                         y_zeros)
    loss_real, acc_real = encoder_discriminator.evaluate(sample_images,
                                                         y_ones)
    d_loss, d_acc = 0.5 * \
        np.add(loss_fake, loss_real), 0.5 * np.add(acc_fake, acc_real)

    encoder_discriminator_loss.append(d_loss)
    encoder_discriminator_accuracy.append(d_acc)

    g_loss, g_acc = encoder_gan.evaluate(sample_states, y_ones)

    encoder_generator_loss.append(g_loss)
    encoder_generator_accuracy.append(g_acc)

    # Display encoded images
    encoded_name = '{} - 2 - Encoded'.format(current_round + 1)
    encoded_sample_images = (encoded_sample_images_scaled + 1) / 2 * 255
    encoded_sample_images = encoded_sample_images[:, :, :, 0]
    image_displayer.display_samples(name=encoded_name,
                                    samples=encoded_sample_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)

    ####################
    # Decode image
    ####################

    # image -> state
    decoded_states = decoder_gan.predict(sample_states)

    # Evaluate
    loss, accuracy = decoder_gan.evaluate(sample_states, sample_states)
    decoder_loss.append(loss)
    decoder_accuracy.append(accuracy)

    # state -> sentence
    decoded_sentences = []
    for decoded_state in decoded_states:

        decoded_state = np.expand_dims(decoded_state, axis=0)
        decoder_input = [decoded_state]

        # Initialize decoder input as a length 1 sentence containing "startofsentence",
        # --> feeding the start token as the first predicted word
        prev_word = np.zeros((1, 1, 1))
        prev_word[0, 0, 0] = word_index["startofsentence"]

        stop_condition = False
        decoded_words = []

        while not stop_condition:
            # 1. predict the next word using decoder model
            logits, states = seq2seq_decoder_model.predict(
                [prev_word] + decoder_input)

            # 2. Update prev_word with the predicted word
            predicted_id = np.argmax(logits[0, 0, :])
            if predicted_id == 0:
                stop_condition = True
                continue
            predicted_word = index_word[predicted_id]
            decoded_words.append(predicted_word)

            # 3. Enable End Condition: (1) if predicted word is "endofsentence" OR
            #                          (2) if translated sentence reached maximum sentence length
            if (predicted_word == 'endofsentence' or len(decoded_words) > max_sequence_length):
                stop_condition = True

            # 4. Update prev_word with the predicted word
            prev_word[0, 0, 0] = predicted_id

            # 5. Update initial_states with the previously predicted word's encoder output
            decoder_input = [states]

        decoded_sentence = " ".join(decoded_words).replace('endofsentence', '')
        decoded_sentences.append(decoded_sentence)

    # Display decoded sentences
    decoded_name = '{} - 3 - Decoded'.format(current_round + 1)
    text_displayer.display_samples(name=decoded_name,
                                   samples=decoded_sentences,
                                   should_display_directly=should_display_directly,
                                   should_save_to_file=should_save_to_file)

    # Calculate BLEU score
    score = bleu_score_calculator.calculate(
        sample_sentences, decoded_sentences)
    bleu_scores.append(score)
    print('BLEU score: {}'.format(score))

diagram_displayer.display_samples(name='Seq2Seq Loss',
                                  samples=seq2seq_loss,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Seq2Seq Accuracy',
                                  samples=seq2seq_accuracy,
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

seq2seq_train_model.save(
    'model/one_layer_sentence_fashion_seq2seq_train_model.h5')
seq2seq_encoder_model.save(
    'model/one_layer_sentence_fashion_seq2seq_encoder_model.h5')
seq2seq_decoder_model.save(
    'model/one_layer_sentence_fashion_seq2seq_decoder_model.h5')

state_encoder.save('model/one_layer_sentence_fashion_state_encoder.h5')
state_decoder.save('model/one_layer_sentence_fashion_state_decoder.h5')
