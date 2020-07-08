from tensorflow.keras.models import load_model

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from generator_models import GeneratorModelCreator
from discriminator_models import DiscriminatorModelCreator
from gan_models import EncoderGanModelCreator, DecoderGanModelCreator

from trainers import EncoderTrainer, DecoderTrainer
from displayers import SampleImageDisplayer, SampleDiagramDisplayer, SampleConfusionMatrixDisplayer, SampleReportDisplayer

from numpy import ones
from numpy import zeros
import numpy as np
import constants

import sys
import os
import re
from PIL import Image

"""
One layer encoding-decoding:
Face -> (Encode) -> Face -> (Decode) -> Face
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

face_images = []
face_labels = []

for face_file in os.scandir(constants.FACE_IMAGE_DATASET_PATH):

    face_file_name = face_file.name
    face_label = re.findall('\d+', face_file_name)[0]
    face_labels.append(face_label)

    face_file_path = constants.FACE_IMAGE_DATASET_PATH + '/' + face_file_name
    face_image = Image.open(face_file_path)
    resized_image = face_image.resize((28, 28))
    image_array = np.asarray(resized_image)
    face_images.append(image_array)

face_images = np.asarray(face_images)
face_labels = np.asarray(face_labels)

unique_labels = np.unique(face_labels)
labels_to_index = {}
for index in range(unique_labels.shape[0]):
    label = unique_labels[index]
    labels_to_index[label] = index
print(labels_to_index)

face_labels = [labels_to_index[label] for label in face_labels]
face_labels = np.asarray(face_labels)

print('Face images shape: {}'.format(face_images.shape))
print('Face labels shape: {}'.format(face_labels.shape))

# Load data
face_images_train, face_images_test, face_labels_train, face_labels_test = train_test_split(face_images,
                                                                                            face_labels,
                                                                                            test_size=0.15)
# Rescale -1 to 1
face_images_train_scaled = (face_images_train / 255.0) * 2 - 1
face_images_test_scaled = (face_images_test / 255.0) * 2 - 1

"""
Create models
"""

# Classifier
try:
    classifier = load_model('model/classifier_face_rgb.h5')
except ImportError:
    print('Unable to load classifier. Please run classifier script first')
    sys.exit()

# Encoder

# Create encoder generator
encoder_generator_creator = GeneratorModelCreator(constants.RGB_INPUT_SHAPE,
                                                  constants.RGB_OUTPUT_SHAPE,
                                                  from_image=True,
                                                  to_image=True,
                                                  activation='tanh')
encoder_generator = encoder_generator_creator.create_model()

# Create encoder discriminator
encoder_discriminator_creator = DiscriminatorModelCreator(
    constants.RGB_INPUT_SHAPE)
encoder_discriminator = encoder_discriminator_creator.create_model()

# Create GAN model to combine encoder generator and discriminator
encoder_gan_creator = EncoderGanModelCreator(encoder_generator,
                                             encoder_discriminator)
encoder_gan = encoder_gan_creator.create_model()

# Decoder

# Create decoder generator
decoder_generator_creator = GeneratorModelCreator(constants.RGB_INPUT_SHAPE,
                                                  constants.RGB_OUTPUT_SHAPE,
                                                  from_image=True,
                                                  to_image=True,
                                                  activation='tanh')
decoder_generator = decoder_generator_creator.create_model()

# Create GAN model to combine encoder generator and decoder generator
decoder_gan_creator = DecoderGanModelCreator(encoder_generator,
                                             decoder_generator)
decoder_gan = decoder_gan_creator.create_model()

"""
Create trainers
"""

# Encoder

# Create encoder trainer
encoder_trainer = EncoderTrainer(encoder_generator,
                                 encoder_discriminator,
                                 encoder_gan,
                                 training_epochs=constants.TRAINING_EPOCHS,
                                 batch_size=constants.BATCH_SIZE)

# Decoder

# Create decoder trainer
decoder_trainer = DecoderTrainer(encoder_generator,
                                 decoder_generator,
                                 decoder_gan,
                                 training_epochs=constants.TRAINING_EPOCHS,
                                 batch_size=constants.BATCH_SIZE)

"""
Start training
"""

image_displayer_rgb = SampleImageDisplayer(row=constants.DISPLAY_ROW,
                                           column=constants.DISPLAY_COLUMN)

diagram_displayer = SampleDiagramDisplayer()

confusion_displayer = SampleConfusionMatrixDisplayer()
report_displayer = SampleReportDisplayer()

encoder_discriminator_loss = []
encoder_discriminator_accuracy = []

encoder_generator_loss = []
encoder_generator_accuracy = []

decoder_loss = []
decoder_accuracy = []

class_loss = []
class_accuracy = []

y_zeros = zeros((constants.DISPLAY_ROW * constants.DISPLAY_COLUMN, 1))
y_ones = ones((constants.DISPLAY_ROW * constants.DISPLAY_COLUMN, 1))

for current_round in range(constants.TOTAL_TRAINING_ROUND):

    print('************************')
    print('Round: {}'.format(current_round + 1))
    print('************************')

    # Train encoder
    encoder_trainer.train(input_data=face_images_train_scaled,
                          exp_output_data=face_images_train_scaled)
    # Train decoder
    decoder_trainer.train(input_data=face_images_train_scaled)

    # Select sample of images
    sample_indexes = np.random.randint(0,
                                       face_images_test.shape[0],
                                       constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    sample_images = face_images_test[sample_indexes]
    sample_labels = face_labels_test[sample_indexes]

    # Display original images
    original_name = 'Original - {}'.format(current_round + 1)
    image_displayer_rgb.display_samples(name=original_name,
                                        samples=sample_images,
                                        should_display_directly=should_display_directly,
                                        should_save_to_file=should_save_to_file,
                                        labels=sample_labels)

    # Evaluate images with labels
    loss_class_original, acc_class_original = classifier.evaluate(sample_images,
                                                                  sample_labels)
    print('Original classification loss: {}, accuracy: {}'.format(
        loss_class_original, acc_class_original))

    # Encode images
    sample_images_scaled = (sample_images / 255.0) * 2 - 1
    encoded_sample_images_scaled = encoder_generator.predict(
        sample_images_scaled)

    # Display encoded images
    encoded_name = 'Encoded - {}'.format(current_round + 1)
    encoded_sample_images = (encoded_sample_images_scaled + 1) / 2 * 255
    encoded_sample_images = encoded_sample_images.astype(int)
    image_displayer_rgb.display_samples(name=encoded_name,
                                        samples=encoded_sample_images,
                                        should_display_directly=should_display_directly,
                                        should_save_to_file=should_save_to_file)

    # Decode images
    decoded_sample_images_scaled = decoder_gan.predict(sample_images_scaled)

    # Display decoded images
    decoded_name = 'Decoded - {}'.format(current_round + 1)
    decoded_sample_images = (decoded_sample_images_scaled + 1) / 2 * 255
    decoded_sample_images = decoded_sample_images.astype(int)

    labels_probs = classifier.predict(decoded_sample_images)
    decoded_sample_labels = []
    for probs in labels_probs:
        label = np.argmax(probs)
        decoded_sample_labels.append(label)

    image_displayer_rgb.display_samples(name=decoded_name,
                                        samples=decoded_sample_images,
                                        should_display_directly=should_display_directly,
                                        should_save_to_file=should_save_to_file,
                                        labels=decoded_sample_labels)

    # Evaluate
    loss_fake, acc_fake = encoder_discriminator.evaluate(encoded_sample_images_scaled,
                                                         y_zeros)
    loss_real, acc_real = encoder_discriminator.evaluate(sample_images_scaled,
                                                         y_ones)
    d_loss, d_acc = 0.5 * \
        np.add(loss_fake, loss_real), 0.5 * np.add(acc_fake, acc_real)

    encoder_discriminator_loss.append(d_loss)
    encoder_discriminator_accuracy.append(d_acc)

    g_loss, g_acc = encoder_gan.evaluate(sample_images_scaled, y_ones)

    encoder_generator_loss.append(g_loss)
    encoder_generator_accuracy.append(g_acc)

    loss, accuracy = decoder_gan.evaluate(
        sample_images_scaled, sample_images_scaled)

    decoder_loss.append(loss)
    decoder_accuracy.append(accuracy)

    # Evaluate images with labels
    loss_class, acc_class = classifier.evaluate(decoded_sample_images,
                                                sample_labels)
    class_loss.append(loss_class)
    class_accuracy.append(acc_class)
    print('Decoded classification loss: {}, accuracy: {}'.format(
        loss_class, acc_class))

    # Calculate recall and precision and f1 score
    confusion = confusion_matrix(sample_labels,
                                 decoded_sample_labels)
    confusion_name = 'Confusion Matrix - {}'.format(current_round + 1)
    confusion_displayer.display_samples(name=confusion_name,
                                        samples=confusion,
                                        should_display_directly=should_display_directly,
                                        should_save_to_file=should_save_to_file)

    classification = classification_report(sample_labels,
                                           decoded_sample_labels)
    report = {
        'classification': classification,
        'loss_class_original': loss_class_original,
        'acc_class_original': acc_class_original,
        'encoder_discriminator_loss': d_loss,
        'encoder_discriminator_accuracy': d_acc,
        'encoder_generator_loss': g_loss,
        'encoder_generator_accuracy': g_acc,
        'decoder_loss': loss,
        'decoder_accuracy': accuracy,
        'loss_class': loss_class,
        'acc_class': acc_class
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

diagram_displayer.display_samples(name='Class Loss',
                                  samples=class_loss,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Class Accuracy',
                                  samples=class_accuracy,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

encoder_generator.save('model/one_layer_face_face_rgb_encoder_generator.h5')
decoder_generator.save('model/one_layer_face_face_rgb_decoder_generator.h5')
