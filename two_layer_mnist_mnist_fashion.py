from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist

from models import ImageGeneratorModelCreator, DiscriminatorModelCreator, EncoderGanModelCreator, DecoderGanModelCreator
from trainers import EncoderTrainer, DecoderTrainer
from displayers import SampleImageDisplayer

import numpy as np
import constants

import sys

"""
Two layer encoding-decoding:
MNIST -> (Encode) -> MNIST -> (Encode) -> Fashion -> (Decode) -> MNIST -> (Decode) -> MNIST 
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
Outer layer models -> Inner layer models -> Outer layer models
"""

""" Encoder - Outer layer """

# Create encoder generator
outer_encoder_generator_creator = ImageGeneratorModelCreator(
    constants.INPUT_SHAPE)
outer_encoder_generator = outer_encoder_generator_creator.create_model()

# Create encoder discriminator
outer_encoder_discriminator_creator = DiscriminatorModelCreator(
    constants.INPUT_SHAPE)
outer_encoder_discriminator = outer_encoder_discriminator_creator.create_model()

# Create GAN model to combine encoder generator and discriminator
outer_encoder_gan_creator = EncoderGanModelCreator(outer_encoder_generator,
                                                   outer_encoder_discriminator)
outer_encoder_gan = outer_encoder_gan_creator.create_model()

""" Encoder - Inner layer """

# Create encoder generator
inner_encoder_generator_creator = ImageGeneratorModelCreator(constants.INPUT_SHAPE)
inner_encoder_generator = inner_encoder_generator_creator.create_model()

# Create encoder discriminator
inner_encoder_discriminator_creator = DiscriminatorModelCreator(
    constants.INPUT_SHAPE)
inner_encoder_discriminator = inner_encoder_discriminator_creator.create_model()

# Create GAN model to combine encoder generator and discriminator
inner_encoder_gan_creator = EncoderGanModelCreator(inner_encoder_generator,
                                                   inner_encoder_discriminator)
inner_encoder_gan = inner_encoder_gan_creator.create_model()

""" Decoder - Inner layer """

# Create decoder generator
inner_decoder_generator_creator = ImageGeneratorModelCreator(constants.INPUT_SHAPE)
inner_decoder_generator = inner_decoder_generator_creator.create_model()

# Create GAN model to combine encoder generator and decoder generator
inner_decoder_gan_creator = DecoderGanModelCreator(inner_encoder_generator,
                                                   inner_decoder_generator)
inner_decoder_gan = inner_decoder_gan_creator.create_model()

""" Decoder - Outer layer """

# Create decoder generator
outer_decoder_generator_creator = ImageGeneratorModelCreator(constants.INPUT_SHAPE)
outer_decoder_generator = outer_decoder_generator_creator.create_model()

# Create GAN model to combine encoder generator and decoder generator
outer_decoder_gan_creator = DecoderGanModelCreator(outer_encoder_generator,
                                                   outer_decoder_generator)
outer_decoder_gan = outer_decoder_gan_creator.create_model()

"""
Create trainers
"""

""" Encoder - Outer layer """

# Create encoder trainer
outer_encoder_trainer = EncoderTrainer(outer_encoder_generator,
                                       outer_encoder_discriminator,
                                       outer_encoder_gan,
                                       training_epochs=constants.TRAINING_EPOCHS,
                                       batch_size=constants.BATCH_SIZE,
                                       input_data=mnist_image_train_scaled,
                                       exp_output_data=mnist_image_train_scaled)

""" Encoder - Inner layer """

# Create encoder trainer
inner_encoder_trainer = EncoderTrainer(inner_encoder_generator,
                                       inner_encoder_discriminator,
                                       inner_encoder_gan,
                                       training_epochs=constants.TRAINING_EPOCHS,
                                       batch_size=constants.BATCH_SIZE,
                                       input_data=mnist_image_train_scaled,
                                       exp_output_data=fashion_image_train_scaled)

""" Decoder - Inner layer """

# Create decoder trainer
inner_decoder_trainer = DecoderTrainer(inner_encoder_generator,
                                       inner_decoder_generator,
                                       inner_decoder_gan,
                                       training_epochs=constants.TRAINING_EPOCHS,
                                       batch_size=constants.BATCH_SIZE,
                                       input_data=mnist_image_train_scaled)

""" Decoder - Outer layer """

# Create decoder trainer
outer_decoder_trainer = DecoderTrainer(outer_encoder_generator,
                                       outer_decoder_generator,
                                       outer_decoder_gan,
                                       training_epochs=constants.TRAINING_EPOCHS,
                                       batch_size=constants.BATCH_SIZE,
                                       input_data=mnist_image_train_scaled)

"""
Start training
"""

image_displayer = SampleImageDisplayer(row=constants.DISPLAY_ROW,
                                       column=constants.DISPLAY_COLUMN)

for current_round in range(constants.TOTAL_TRAINING_ROUND):

    print('************************')
    print('Round: {}'.format(current_round + 1))
    print('************************')

    # Train: outer encoder -> inner encoder -> inner decoder -> outer decoder
    outer_encoder_trainer.train_model()
    inner_encoder_trainer.train_model()
    inner_decoder_trainer.train_model()
    outer_decoder_trainer.train_model()

    # Select sample of images
    original_indexes = np.random.randint(0,
                                         mnist_image_test.shape[0],
                                         constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    original_images = mnist_image_test[original_indexes]

    # Display original images
    original_name = '{} - 1 - Original'.format(current_round + 1)
    image_displayer.display_samples(name=original_name,
                                    samples=original_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)

    """ Encoder - Outer layer """

    # Encode images
    original_images_scaled = (original_images / 255.0) * 2 - 1
    outer_encoded_images_scaled = outer_encoder_generator.predict(
        original_images_scaled)

    # Display encoded images
    outer_encoded_name = '{} - 2 - Outer - Encoded'.format(current_round + 1)
    outer_encoded_images = (outer_encoded_images_scaled + 1) / 2 * 255
    outer_encoded_images = outer_encoded_images[:, :, :, 0]
    image_displayer.display_samples(name=outer_encoded_name,
                                    samples=outer_encoded_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)

    """ Encoder - Inner layer """

    # Encode images
    inner_encoded_images_scaled = inner_encoder_generator.predict(
        outer_encoded_images_scaled)

    # Display encoded images
    inner_encoded_name = '{} - 3 - Inner - Encoded'.format(current_round + 1)
    inner_encoded_images = (inner_encoded_images_scaled + 1) / 2 * 255
    inner_encoded_images = inner_encoded_images[:, :, :, 0]
    image_displayer.display_samples(name=inner_encoded_name,
                                    samples=inner_encoded_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)

    """ Decoder - Inner layer """

    # Decode images
    inner_decoded_images_scaled = inner_decoder_generator.predict(
        inner_encoded_images_scaled)

    # Display decoded images
    inner_decoded_name = '{} - 4 - Inner - Decoded'.format(current_round + 1)
    inner_decoded_images = (inner_decoded_images_scaled + 1) / 2 * 255
    inner_decoded_images = inner_decoded_images[:, :, :, 0]
    image_displayer.display_samples(name=inner_decoded_name,
                                    samples=inner_decoded_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)

    """ Decoder - Outer layer """

    # Decode images
    outer_decoded_images_scaled = outer_decoder_generator.predict(
        inner_decoded_images_scaled)

    # Display decoded images
    outer_decoded_name = '{} - 5 - Outer - Decoded'.format(current_round + 1)
    outer_decoded_images = (outer_decoded_images_scaled + 1) / 2 * 255
    outer_decoded_images = outer_decoded_images[:, :, :, 0]
    image_displayer.display_samples(name=outer_decoded_name,
                                    samples=outer_decoded_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)
