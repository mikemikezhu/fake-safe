from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

from generator_models import ImageGeneratorModelCreator
from discriminator_models import DiscriminatorModelCreator
from gan_models import EncoderGanModelCreator, DecoderGanModelCreator

from trainers import EncoderTrainer, DecoderTrainer
from displayers import SampleImageDisplayer

from numpy import ones
from numpy import zeros
import numpy as np
import constants

import matplotlib.pyplot as plt
import sys
import os
from PIL import Image
from zipfile import ZipFile

"""
One layer encoding-decoding:
Face -> (Encode) -> MNIST -> (Decode) -> Face
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
for face_file in os.scandir(constants.FACE_IMAGE_DATASET_PATH):
    face_file_path = constants.FACE_IMAGE_DATASET_PATH + '/' + face_file.name
    face_image = Image.open(face_file_path)
    grey_image = face_image.convert('L')
    resized_image = grey_image.resize((28, 28))
    image_array = np.asarray(resized_image)
    face_images.append(image_array)
face_images = np.asarray(face_images)

# Load data
(mnist_image_train, _), (mnist_image_test, _) = mnist.load_data()
face_images_train, face_images_test = train_test_split(
    face_images, test_size=0.15)

# Rescale -1 to 1
mnist_image_train_scaled = (mnist_image_train / 255.0) * 2 - 1
mnist_image_test_scaled = (mnist_image_test / 255.0) * 2 - 1

face_images_train_scaled = (face_images_train / 255.0) * 2 - 1
face_images_test_scaled = (face_images_test / 255.0) * 2 - 1

"""
Create models
"""

# Encoder

# Create encoder generator
encoder_generator_creator = ImageGeneratorModelCreator(constants.INPUT_SHAPE,
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

# Create decoder generator
decoder_generator_creator = ImageGeneratorModelCreator(constants.INPUT_SHAPE,
                                                       constants.OUTPUT_SHAPE)
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
                                 batch_size=constants.BATCH_SIZE,
                                 input_data=face_images_train_scaled,
                                 exp_output_data=mnist_image_train_scaled)

# Decoder

# Create decoder trainer
decoder_trainer = DecoderTrainer(encoder_generator,
                                 decoder_generator,
                                 decoder_gan,
                                 training_epochs=constants.TRAINING_EPOCHS,
                                 batch_size=constants.BATCH_SIZE,
                                 input_data=face_images_train_scaled)

"""
Start training
"""

image_displayer = SampleImageDisplayer(row=constants.DISPLAY_ROW,
                                       column=constants.DISPLAY_COLUMN)

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

    # Train encoder
    encoder_trainer.train_model()
    # Train decoder
    decoder_trainer.train_model()

    # Select sample of images
    sample_indexes = np.random.randint(0,
                                       face_images_test.shape[0],
                                       constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    sample_images = face_images_test[sample_indexes]

    mnist_image_indexes = np.random.randint(0,
                                            mnist_image_test_scaled.shape[0],
                                            constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    sample_mnist_image = mnist_image_test_scaled[mnist_image_indexes]

    # Display original images
    original_name = 'Original - {}'.format(current_round + 1)
    image_displayer.display_samples(name=original_name,
                                    samples=sample_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)

    # Encode images
    sample_images_scaled = (sample_images / 255.0) * 2 - 1
    encoded_sample_images_scaled = encoder_generator.predict(
        sample_images_scaled)

    # Display encoded images
    encoded_name = 'Encoded - {}'.format(current_round + 1)
    encoded_sample_images = (encoded_sample_images_scaled + 1) / 2 * 255
    encoded_sample_images = encoded_sample_images[:, :, :, 0]
    image_displayer.display_samples(name=encoded_name,
                                    samples=encoded_sample_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)

    # Decode images
    decoded_sample_images_scaled = decoder_gan.predict(sample_images_scaled)

    # Display decoded images
    decoded_name = 'Decoded - {}'.format(current_round + 1)
    decoded_sample_images = (decoded_sample_images_scaled + 1) / 2 * 255
    decoded_sample_images = decoded_sample_images[:, :, :, 0]
    image_displayer.display_samples(name=decoded_name,
                                    samples=decoded_sample_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)

    # Evaluate
    loss_fake, acc_fake = encoder_discriminator.evaluate(encoded_sample_images_scaled,
                                                         y_zeros)
    loss_real, acc_real = encoder_discriminator.evaluate(sample_mnist_image,
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

plt.title('Encoder Discriminator Loss')
plt.plot(encoder_discriminator_loss)
plt.savefig('output/encoder_discriminator_loss.png')
plt.close()

plt.title('Encoder Discriminator Accuracy')
plt.plot(encoder_discriminator_accuracy)
plt.savefig('output/encoder_discriminator_accuracy.png')
plt.close()

plt.title('Encoder Generator Loss')
plt.plot(encoder_generator_loss)
plt.savefig('output/encoder_generator_loss.png')
plt.close()

plt.title('Encoder Generator Accuracy')
plt.plot(encoder_generator_accuracy)
plt.savefig('output/encoder_generator_accuracy.png')
plt.close()

plt.title('Decoder Loss')
plt.plot(decoder_loss)
plt.savefig('output/decoder_loss.png')
plt.close()

plt.title('Decoder Accuracy')
plt.plot(decoder_accuracy)
plt.savefig('output/decoder_accuracy.png')
plt.close()