from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist

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

"""
Three layer encoding-decoding:
MNIST -> (Encode) -> MNIST -> (Encode) -> Fashion -> (Encode) -> Fashion ->
(Decode) -> Fashion -> (Decode) -> MNIST -> (Decode) -> MNIST 
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
Outer layer models -> Middle layer models -> Inner layer models -> 
Middle layer models -> Outer layer models
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

""" Encoder - Middle layer """

# Create encoder generator
mid_encoder_generator_creator = ImageGeneratorModelCreator(
    constants.INPUT_SHAPE)
mid_encoder_generator = mid_encoder_generator_creator.create_model()

# Create encoder discriminator
mid_encoder_discriminator_creator = DiscriminatorModelCreator(
    constants.INPUT_SHAPE)
mid_encoder_discriminator = mid_encoder_discriminator_creator.create_model()

# Create GAN model to combine encoder generator and discriminator
mid_encoder_gan_creator = EncoderGanModelCreator(mid_encoder_generator,
                                                 mid_encoder_discriminator)
mid_encoder_gan = mid_encoder_gan_creator.create_model()

""" Encoder - Inner layer """

# Create encoder generator
inner_encoder_generator_creator = ImageGeneratorModelCreator(
    constants.INPUT_SHAPE)
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
inner_decoder_generator_creator = ImageGeneratorModelCreator(
    constants.INPUT_SHAPE)
inner_decoder_generator = inner_decoder_generator_creator.create_model()

# Create GAN model to combine encoder generator and decoder generator
inner_decoder_gan_creator = DecoderGanModelCreator(inner_encoder_generator,
                                                   inner_decoder_generator)
inner_decoder_gan = inner_decoder_gan_creator.create_model()

""" Decoder - Middle layer """

# Create decoder generator
mid_decoder_generator_creator = ImageGeneratorModelCreator(
    constants.INPUT_SHAPE)
mid_decoder_generator = mid_decoder_generator_creator.create_model()

# Create GAN model to combine encoder generator and decoder generator
mid_decoder_gan_creator = DecoderGanModelCreator(mid_encoder_generator,
                                                 mid_decoder_generator)
mid_decoder_gan = mid_decoder_gan_creator.create_model()

""" Decoder - Outer layer """

# Create decoder generator
outer_decoder_generator_creator = ImageGeneratorModelCreator(
    constants.INPUT_SHAPE)
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

""" Encoder - Middle layer """

# Create encoder trainer
mid_encoder_trainer = EncoderTrainer(mid_encoder_generator,
                                     mid_encoder_discriminator,
                                     mid_encoder_gan,
                                     training_epochs=constants.TRAINING_EPOCHS,
                                     batch_size=constants.BATCH_SIZE,
                                     input_data=mnist_image_train_scaled,
                                     exp_output_data=fashion_image_train_scaled)

""" Encoder - Inner layer """

# Create encoder trainer
inner_encoder_trainer = EncoderTrainer(inner_encoder_generator,
                                       inner_encoder_discriminator,
                                       inner_encoder_gan,
                                       training_epochs=constants.TRAINING_EPOCHS,
                                       batch_size=constants.BATCH_SIZE,
                                       input_data=fashion_image_train_scaled,
                                       exp_output_data=fashion_image_train_scaled)

""" Decoder - Inner layer """

# Create decoder trainer
inner_decoder_trainer = DecoderTrainer(inner_encoder_generator,
                                       inner_decoder_generator,
                                       inner_decoder_gan,
                                       training_epochs=constants.TRAINING_EPOCHS,
                                       batch_size=constants.BATCH_SIZE,
                                       input_data=fashion_image_train_scaled)

""" Decoder - Middle layer """

# Create decoder trainer
mid_decoder_trainer = DecoderTrainer(mid_encoder_generator,
                                     mid_decoder_generator,
                                     mid_decoder_gan,
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

encoder_discriminator_loss_outer = []
encoder_discriminator_accuracy_outer = []

encoder_discriminator_loss_mid = []
encoder_discriminator_accuracy_mid = []

encoder_discriminator_loss_inner = []
encoder_discriminator_accuracy_inner = []

encoder_generator_loss_outer = []
encoder_generator_accuracy_outer = []

encoder_generator_loss_mid = []
encoder_generator_accuracy_mid = []

encoder_generator_loss_inner = []
encoder_generator_accuracy_inner = []

decoder_loss_outer = []
decoder_accuracy_outer = []

decoder_loss_mid = []
decoder_accuracy_mid = []

decoder_loss_inner = []
decoder_accuracy_inner = []

y_zeros = zeros((constants.DISPLAY_ROW * constants.DISPLAY_COLUMN, 1))
y_ones = ones((constants.DISPLAY_ROW * constants.DISPLAY_COLUMN, 1))

for current_round in range(constants.TOTAL_TRAINING_ROUND):

    print('************************')
    print('Round: {}'.format(current_round + 1))
    print('************************')

    # Train: outer encoder -> inner encoder -> middle encoder ->
    # inner decoder -> middle decoder -> outer decoder
    outer_encoder_trainer.train_model()
    mid_encoder_trainer.train_model()
    inner_encoder_trainer.train_model()
    inner_decoder_trainer.train_model()
    mid_decoder_trainer.train_model()
    outer_decoder_trainer.train_model()

    # Select sample of images
    original_indexes = np.random.randint(0,
                                         mnist_image_test.shape[0],
                                         constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    original_images = mnist_image_test[original_indexes]

    fashion_indexes = np.random.randint(0,
                                        fashion_image_test_scaled.shape[0],
                                        constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    fashion_images = fashion_image_test_scaled[fashion_indexes]

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

    # Evaluate
    loss_fake_outer, acc_fake_outer = outer_encoder_discriminator.evaluate(outer_encoded_images_scaled,
                                                                           y_zeros)
    loss_real_outer, acc_real_outer = outer_encoder_discriminator.evaluate(original_images_scaled,
                                                                           y_ones)
    d_loss_outer, d_acc_outer = 0.5 * \
        np.add(loss_fake_outer, loss_real_outer), 0.5 * \
        np.add(acc_fake_outer, acc_real_outer)

    encoder_discriminator_loss_outer.append(d_loss_outer)
    encoder_discriminator_accuracy_outer.append(d_acc_outer)

    g_loss_outer, g_acc_outer = outer_encoder_gan.evaluate(original_images_scaled,
                                                           y_ones)

    encoder_generator_loss_outer.append(g_loss_outer)
    encoder_generator_accuracy_outer.append(g_acc_outer)

    """ Encoder - Middle layer """

    # Encode images
    mid_encoded_images_scaled = mid_encoder_generator.predict(
        outer_encoded_images_scaled)

    # Display encoded images
    mid_encoded_name = '{} - 3 - Middle - Encoded'.format(current_round + 1)
    mid_encoded_images = (mid_encoded_images_scaled + 1) / 2 * 255
    mid_encoded_images = mid_encoded_images[:, :, :, 0]
    image_displayer.display_samples(name=mid_encoded_name,
                                    samples=mid_encoded_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)

    # Evaluate
    loss_fake_mid, acc_fake_mid = mid_encoder_discriminator.evaluate(mid_encoded_images_scaled,
                                                                     y_zeros)
    loss_real_mid, acc_real_mid = mid_encoder_discriminator.evaluate(fashion_images,
                                                                     y_ones)
    d_loss_mid, d_acc_mid = 0.5 * \
        np.add(loss_fake_mid, loss_real_mid), 0.5 * \
        np.add(acc_fake_mid, acc_real_mid)

    encoder_discriminator_loss_mid.append(d_loss_mid)
    encoder_discriminator_accuracy_mid.append(d_acc_mid)

    g_loss_mid, g_acc_mid = mid_encoder_gan.evaluate(original_images_scaled,
                                                     y_ones)

    encoder_generator_loss_mid.append(g_loss_mid)
    encoder_generator_accuracy_mid.append(g_acc_mid)

    """ Encoder - Inner layer """

    # Encode images
    inner_encoded_images_scaled = inner_encoder_generator.predict(
        mid_encoded_images_scaled)

    # Display encoded images
    inner_encoded_name = '{} - 4 - Inner - Encoded'.format(current_round + 1)
    inner_encoded_images = (inner_encoded_images_scaled + 1) / 2 * 255
    inner_encoded_images = inner_encoded_images[:, :, :, 0]
    image_displayer.display_samples(name=inner_encoded_name,
                                    samples=inner_encoded_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)

    # Evaluate
    loss_fake_inner, acc_fake_inner = inner_encoder_discriminator.evaluate(inner_encoded_images_scaled,
                                                                           y_zeros)
    loss_real_inner, acc_real_inner = inner_encoder_discriminator.evaluate(fashion_images,
                                                                           y_ones)
    d_loss_inner, d_acc_inner = 0.5 * \
        np.add(loss_fake_inner, loss_real_inner), 0.5 * \
        np.add(acc_fake_inner, acc_real_inner)

    encoder_discriminator_loss_inner.append(d_loss_inner)
    encoder_discriminator_accuracy_inner.append(d_acc_inner)

    g_loss_inner, g_acc_inner = inner_encoder_gan.evaluate(fashion_images,
                                                           y_ones)

    encoder_generator_loss_inner.append(g_loss_inner)
    encoder_generator_accuracy_inner.append(g_acc_inner)

    """ Decoder - Inner layer """

    # Decode images
    inner_decoded_images_scaled = inner_decoder_generator.predict(
        inner_encoded_images_scaled)

    # Display decoded images
    inner_decoded_name = '{} - 5 - Inner - Decoded'.format(current_round + 1)
    inner_decoded_images = (inner_decoded_images_scaled + 1) / 2 * 255
    inner_decoded_images = inner_decoded_images[:, :, :, 0]
    image_displayer.display_samples(name=inner_decoded_name,
                                    samples=inner_decoded_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)

    # Evaluate
    loss_inner, accuracy_inner = inner_decoder_gan.evaluate(fashion_images,
                                                            fashion_images)

    decoder_loss_inner.append(loss_inner)
    decoder_accuracy_inner.append(accuracy_inner)

    """ Decoder - Middle layer """

    # Decode images
    mid_decoded_images_scaled = mid_decoder_generator.predict(
        inner_decoded_images_scaled)

    # Display decoded images
    mid_decoded_name = '{} - 6 - Middle - Decoded'.format(current_round + 1)
    mid_decoded_images = (mid_decoded_images_scaled + 1) / 2 * 255
    mid_decoded_images = mid_decoded_images[:, :, :, 0]
    image_displayer.display_samples(name=mid_decoded_name,
                                    samples=mid_decoded_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)
    # Evaluate
    loss_mid, accuracy_mid = mid_decoder_gan.evaluate(original_images_scaled,
                                                      original_images_scaled)

    decoder_loss_mid.append(loss_mid)
    decoder_accuracy_mid.append(accuracy_mid)

    """ Decoder - Outer layer """

    # Decode images
    outer_decoded_images_scaled = outer_decoder_generator.predict(
        mid_decoded_images_scaled)

    # Display decoded images
    outer_decoded_name = '{} - 7 - Outer - Decoded'.format(current_round + 1)
    outer_decoded_images = (outer_decoded_images_scaled + 1) / 2 * 255
    outer_decoded_images = outer_decoded_images[:, :, :, 0]
    image_displayer.display_samples(name=outer_decoded_name,
                                    samples=outer_decoded_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file)

    # Evaluate
    loss_outer, accuracy_outer = outer_decoder_gan.evaluate(original_images_scaled,
                                                            original_images_scaled)

    decoder_loss_outer.append(loss_outer)
    decoder_accuracy_outer.append(accuracy_outer)

plt.title('Outer Encoder Discriminator Loss')
plt.plot(encoder_discriminator_loss_outer)
plt.savefig('output/encoder_discriminator_loss_outer.png')
plt.close()

plt.title('Outer Encoder Discriminator Accuracy')
plt.plot(encoder_discriminator_accuracy_outer)
plt.savefig('output/encoder_discriminator_accuracy_outer.png')
plt.close()

plt.title('Middle Encoder Discriminator Loss')
plt.plot(encoder_discriminator_loss_mid)
plt.savefig('output/encoder_discriminator_loss_mid.png')
plt.close()

plt.title('Middle Encoder Discriminator Accuracy')
plt.plot(encoder_discriminator_accuracy_mid)
plt.savefig('output/encoder_discriminator_accuracy_mid.png')
plt.close()

plt.title('Inner Encoder Discriminator Loss')
plt.plot(encoder_discriminator_loss_inner)
plt.savefig('output/encoder_discriminator_loss_inner.png')
plt.close()

plt.title('Inner Encoder Discriminator Accuracy')
plt.plot(encoder_discriminator_accuracy_inner)
plt.savefig('output/encoder_discriminator_accuracy_inner.png')
plt.close()

plt.title('Outer Encoder Generator Loss')
plt.plot(encoder_generator_loss_outer)
plt.savefig('output/encoder_generator_loss_outer.png')
plt.close()

plt.title('Outer Encoder Generator Accuracy')
plt.plot(encoder_generator_accuracy_outer)
plt.savefig('output/encoder_generator_accuracy_outer.png')
plt.close()

plt.title('Middle Encoder Generator Loss')
plt.plot(encoder_generator_loss_mid)
plt.savefig('output/encoder_generator_loss_mid.png')
plt.close()

plt.title('Middle Encoder Generator Accuracy')
plt.plot(encoder_generator_accuracy_mid)
plt.savefig('output/encoder_generator_accuracy_mid.png')
plt.close()

plt.title('Inner Encoder Generator Loss')
plt.plot(encoder_generator_loss_inner)
plt.savefig('output/encoder_generator_loss_inner.png')
plt.close()

plt.title('Inner Encoder Generator Accuracy')
plt.plot(encoder_generator_accuracy_inner)
plt.savefig('output/encoder_generator_accuracy_inner.png')
plt.close()

plt.title('Outer Decoder Loss')
plt.plot(decoder_loss_outer)
plt.savefig('output/decoder_loss_outer.png')
plt.close()

plt.title('Outer Decoder Accuracy')
plt.plot(decoder_accuracy_outer)
plt.savefig('output/decoder_accuracy_outer.png')
plt.close()

plt.title('Middle Decoder Loss')
plt.plot(decoder_loss_mid)
plt.savefig('output/decoder_loss_mid.png')
plt.close()

plt.title('Middle Decoder Accuracy')
plt.plot(decoder_accuracy_mid)
plt.savefig('output/decoder_accuracy_mid.png')
plt.close()

plt.title('Inner Decoder Loss')
plt.plot(decoder_loss_inner)
plt.savefig('output/decoder_loss_inner.png')
plt.close()

plt.title('Inner Decoder Accuracy')
plt.plot(decoder_accuracy_inner)
plt.savefig('output/decoder_accuracy_inner.png')
plt.close()
