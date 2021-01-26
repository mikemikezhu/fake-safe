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
from displayers import SampleTextDisplayer, SampleImageDisplayer, SampleDiagramDisplayer, SampleConfusionMatrixDisplayer, SampleReportDisplayer

from imblearn.over_sampling import RandomOverSampler

from numpy import ones
from numpy import zeros
import numpy as np
import re
import os
import fakesafe_constants as constants

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

# Load text data
with open(constants.TEXT_2_IMAGE_DATASET_PATH, 'r') as data_file:
    lines = data_file.read().split('\n')

words = []
for line in lines:
    line = line.strip()  # Remove "\n" and empty space
    line = line.lower()  # Lower all characters
    line = re.sub('[^a-z ]+', '', line)  # Remove non-alphabet characters
    line = re.sub(' +', ' ', line)  # Remove extra empty space
    words += [word for word in line.split(' ') if word]

# Since the dataset does not have enough corpus to train the model
# We only select some of the most frequent words to demonstrate out idea
words_dict = {}
for word in words:
    count = words_dict.get(word)
    if not count:
        count = 0
    count += 1
    words_dict[word] = count

sorted_words_tuple = sorted(
    words_dict.items(), key=lambda x: x[1], reverse=True)
sorted_words_tuple = sorted_words_tuple[:100]
sorted_words = [tuple[0] for tuple in sorted_words_tuple]

words = [word for word in words if word in sorted_words]
oversample = RandomOverSampler(sampling_strategy='minority')
words = np.asarray(words)
print('Words shape before over sampling: {}'.format(words.shape))

words = words.reshape(-1, 1)
words, words = oversample.fit_resample(words, words)
print('Words shape after over sampling: {}'.format(words.shape))

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
word_embeddings = []
for word in words:
    embedding_vector = original_embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_index[word] = embedding_vector
        word_embeddings.append(embedding_vector)
    else:
        print('Word not found: {}'.format(word))

word_embeddings = np.asarray(word_embeddings)
print(word_embeddings.shape)

# Create train and test set
words_train, words_test = train_test_split(words,
                                           test_size=0.15,
                                           random_state=1)
word_embeddings_train, word_embeddings_test = train_test_split(word_embeddings,
                                                               test_size=0.15,
                                                               random_state=1)

print('Train shape: {}'.format(words_train.shape))
print('Test shape: {}'.format(words_test.shape))

print('Embeddings train shape: {}'.format(word_embeddings_train.shape))
print('Embeddings test shape: {}'.format(word_embeddings_test.shape))

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
confusion_displayer = SampleConfusionMatrixDisplayer()
report_displayer = SampleReportDisplayer()
image_displayer = SampleImageDisplayer(row=constants.DISPLAY_ROW,
                                       column=constants.DISPLAY_COLUMN,
                                       cmap='gray')

encoder_discriminator_loss = []
encoder_discriminator_accuracy = []

encoder_generator_loss = []
encoder_generator_accuracy = []

decoder_loss = []
decoder_accuracy = []

y_zeros = zeros((constants.DISPLAY_ROW * constants.DISPLAY_COLUMN, 1))
y_ones = ones((constants.DISPLAY_ROW * constants.DISPLAY_COLUMN, 1))

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
                                       word_embeddings_test.shape[0],
                                       constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    sample_words = words_test[sample_indexes]
    sample_embeddings = word_embeddings_test[sample_indexes]

    # Select a random batch of images
    image_indexes = np.random.randint(0,
                                      fashion_image_train_scaled.shape[0],
                                      constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    sample_images = fashion_image_train_scaled[image_indexes]

    # Display original corpus
    original_name = '{} - 1 - Original'.format(current_round + 1)
    text_displayer.display_samples(name=original_name,
                                   samples=sample_words,
                                   should_display_directly=should_display_directly,
                                   should_save_to_file=should_save_to_file)

    # Encode images
    encoded_sample_images_scaled = word_encoder.predict(sample_embeddings)

    # Evaluate
    loss_fake, acc_fake = encoder_discriminator.evaluate(encoded_sample_images_scaled,
                                                         y_zeros)
    loss_real, acc_real = encoder_discriminator.evaluate(sample_images,
                                                         y_ones)
    d_loss, d_acc = 0.5 * \
        np.add(loss_fake, loss_real), 0.5 * np.add(acc_fake, acc_real)

    encoder_discriminator_loss.append(d_loss)
    encoder_discriminator_accuracy.append(d_acc)

    g_loss, g_acc = encoder_gan.evaluate(sample_embeddings, y_ones)

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

    # Decode images to word
    loss, accuracy = decoder_gan.evaluate(sample_embeddings, sample_embeddings)
    decoder_loss.append(loss)
    decoder_accuracy.append(accuracy)

    print('Decode loss: {}, accuracy: {}'.format(loss, accuracy))
    decoded_embeddings = decoder_gan.predict(sample_embeddings)

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

    print('Sample words:')
    print(sample_words)
    print('Decoded words:')
    print(decoded_words)

    # Display decoded word
    decoded_name = '{} - 3 - Decoded'.format(current_round + 1)
    text_displayer.display_samples(name=decoded_name,
                                   samples=decoded_words,
                                   should_display_directly=should_display_directly,
                                   should_save_to_file=should_save_to_file)

    # Calculate recall and precision and f1 score
    confusion = confusion_matrix(sample_words,
                                 decoded_words)
    confusion_name = 'Confusion Matrix - {}'.format(current_round + 1)
    confusion_displayer.display_samples(name=confusion_name,
                                        samples=confusion,
                                        should_display_directly=should_display_directly,
                                        should_save_to_file=should_save_to_file)

    classification = classification_report(sample_words,
                                           decoded_words)
    report = {
        'classification': classification,
        'encoder_discriminator_loss': d_loss,
        'encoder_discriminator_accuracy': d_acc,
        'encoder_generator_loss': g_loss,
        'encoder_generator_accuracy': g_acc,
        'decoder_loss': loss,
        'decoder_accuracy': accuracy
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

print('Best accuracy: {}'.format(max(decoder_accuracy)))

word_encoder.save(
    'model/one_layer_word_fashion_embedding_encoder_generator.h5')
word_decoder.save(
    'model/one_layer_word_fashion_embedding_decoder_generator.h5')
