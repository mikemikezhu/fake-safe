from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist

from generator_models import ImageGeneratorModelCreator
from discriminator_models import DiscriminatorModelCreator
from gan_models import EncoderGanModelCreator, DecoderGanModelCreator

from trainers import EncoderTrainer, DecoderTrainer
from displayers import SampleImageDisplayer

import numpy as np
import constants

import sys

"""
One layer encoding-decoding:
Fashion -> (Encode) -> MNIST -> (Decode) -> Fashion
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

# Load data
(mnist_image_train, _), (mnist_image_test, _) = mnist.load_data()
(fashion_image_train, _), (fashion_image_test, _) = fashion_mnist.load_data()

# Rescale -1 to 1
mnist_image_train_scaled = (mnist_image_train / 255.0) * 2 - 1
mnist_image_test_scaled = (mnist_image_test / 255.0) * 2 - 1
fashion_image_train_scaled = (fashion_image_train / 255.0) * 2 - 1
fashion_image_test_scaled = (fashion_image_test / 255.0) * 2 - 1

"""
Create models
"""

# Encoder

# Create encoder generator
encoder_generator_creator = ImageGeneratorModelCreator(constants.INPUT_SHAPE)
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
decoder_generator_creator = ImageGeneratorModelCreator(constants.INPUT_SHAPE)
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
                                 input_data=fashion_image_train_scaled,
                                 exp_output_data=mnist_image_train_scaled)

# Decoder

# Create decoder trainer
decoder_trainer = DecoderTrainer(encoder_generator,
                                 decoder_generator,
                                 decoder_gan,
                                 training_epochs=constants.TRAINING_EPOCHS,
                                 batch_size=constants.BATCH_SIZE,
                                 input_data=fashion_image_train_scaled)

"""
Start training
"""

image_displayer = SampleImageDisplayer(row=constants.DISPLAY_ROW,
                                       column=constants.DISPLAY_COLUMN)

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
                                       fashion_image_test.shape[0],
                                       constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    sample_images = fashion_image_test[sample_indexes]

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
