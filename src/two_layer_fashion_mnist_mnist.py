from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from generator_models import ImageGeneratorModelCreator
from discriminator_models import DiscriminatorModelCreator
from gan_models import EncoderGanModelCreator, DecoderGanModelCreator

from trainers import EncoderTrainer, DecoderTrainer
from displayers import SampleImageDisplayer, SampleDiagramDisplayer, SampleConfusionMatrixDisplayer, SampleReportDisplayer

from numpy import ones
from numpy import zeros
import numpy as np
import constants

import sys

"""
Two layer encoding-decoding:
Fashion -> (Encode) -> MNIST -> (Encode) -> MNIST -> (Decode) -> MNIST -> (Decode) -> Fashion 
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
(fashion_image_train, fashion_label_train), (fashion_image_test,
                                             fashion_label_test) = fashion_mnist.load_data()

# Rescale -1 to 1
mnist_image_train_scaled = (mnist_image_train / 255.0) * 2 - 1
mnist_image_test_scaled = (mnist_image_test / 255.0) * 2 - 1
fashion_image_train_scaled = (fashion_image_train / 255.0) * 2 - 1
fashion_image_test_scaled = (fashion_image_test / 255.0) * 2 - 1

"""
Create models
Outer layer models -> Inner layer models -> Outer layer models
"""

# Classifier
try:
    classifier = load_model('model/classifier_fashion.h5')
except ImportError:
    print('Unable to load classifier. Please run classifier script first')
    sys.exit()

""" Encoder - Outer layer """

# Create encoder generator
outer_encoder_generator_creator = ImageGeneratorModelCreator(constants.INPUT_SHAPE,
                                                             constants.OUTPUT_SHAPE)
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
inner_encoder_generator_creator = ImageGeneratorModelCreator(constants.INPUT_SHAPE,
                                                             constants.OUTPUT_SHAPE)
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
inner_decoder_generator_creator = ImageGeneratorModelCreator(constants.INPUT_SHAPE,
                                                             constants.OUTPUT_SHAPE)
inner_decoder_generator = inner_decoder_generator_creator.create_model()

# Create GAN model to combine encoder generator and decoder generator
inner_decoder_gan_creator = DecoderGanModelCreator(inner_encoder_generator,
                                                   inner_decoder_generator)
inner_decoder_gan = inner_decoder_gan_creator.create_model()

""" Decoder - Outer layer """

# Create decoder generator
outer_decoder_generator_creator = ImageGeneratorModelCreator(constants.INPUT_SHAPE,
                                                             constants.OUTPUT_SHAPE)
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
                                       batch_size=constants.BATCH_SIZE)

""" Encoder - Inner layer """

# Create encoder trainer
inner_encoder_trainer = EncoderTrainer(inner_encoder_generator,
                                       inner_encoder_discriminator,
                                       inner_encoder_gan,
                                       training_epochs=constants.TRAINING_EPOCHS,
                                       batch_size=constants.BATCH_SIZE)

""" Decoder - Inner layer """

# Create decoder trainer
inner_decoder_trainer = DecoderTrainer(inner_encoder_generator,
                                       inner_decoder_generator,
                                       inner_decoder_gan,
                                       training_epochs=constants.TRAINING_EPOCHS,
                                       batch_size=constants.BATCH_SIZE)

""" Decoder - Outer layer """

# Create decoder trainer
outer_decoder_trainer = DecoderTrainer(outer_encoder_generator,
                                       outer_decoder_generator,
                                       outer_decoder_gan,
                                       training_epochs=constants.TRAINING_EPOCHS,
                                       batch_size=constants.BATCH_SIZE)

"""
Start training
"""

image_displayer = SampleImageDisplayer(row=constants.DISPLAY_ROW,
                                       column=constants.DISPLAY_COLUMN,
                                       cmap='gray')

diagram_displayer = SampleDiagramDisplayer()

confusion_displayer = SampleConfusionMatrixDisplayer()
report_displayer = SampleReportDisplayer()

encoder_discriminator_loss_outer = []
encoder_discriminator_accuracy_outer = []

encoder_discriminator_loss_inner = []
encoder_discriminator_accuracy_inner = []

encoder_generator_loss_outer = []
encoder_generator_accuracy_outer = []

encoder_generator_loss_inner = []
encoder_generator_accuracy_inner = []

decoder_loss_outer = []
decoder_accuracy_outer = []

decoder_loss_inner = []
decoder_accuracy_inner = []

class_loss = []
class_accuracy = []

y_zeros = zeros((constants.DISPLAY_ROW * constants.DISPLAY_COLUMN, 1))
y_ones = ones((constants.DISPLAY_ROW * constants.DISPLAY_COLUMN, 1))

for current_round in range(constants.TOTAL_TRAINING_ROUND):

    print('************************')
    print('Round: {}'.format(current_round + 1))
    print('************************')

    # Train: outer encoder -> inner encoder -> inner decoder -> outer decoder
    outer_encoder_trainer.train(input_data=fashion_image_train_scaled,
                                exp_output_data=mnist_image_train_scaled)
    inner_encoder_trainer.train(input_data=mnist_image_train_scaled,
                                exp_output_data=mnist_image_train_scaled)
    inner_decoder_trainer.train(input_data=mnist_image_train_scaled)
    outer_decoder_trainer.train(input_data=fashion_image_train_scaled)

    # Select sample of images
    original_indexes = np.random.randint(0,
                                         fashion_image_test.shape[0],
                                         constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    original_images = fashion_image_test[original_indexes]
    original_labels = fashion_label_test[original_indexes]

    mnist_indexes = np.random.randint(0,
                                      mnist_image_test_scaled.shape[0],
                                      constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
    mnist_images = mnist_image_test_scaled[mnist_indexes]

    # Display original images
    original_name = '{} - 1 - Original'.format(current_round + 1)
    image_displayer.display_samples(name=original_name,
                                    samples=original_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file,
                                    labels=original_labels)

    # Evaluate images with labels
    loss_class_original, acc_class_original = classifier.evaluate(original_images,
                                                                  original_labels)
    print('Original classification loss: {}, accuracy: {}'.format(
        loss_class_original, acc_class_original))

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
    loss_real_outer, acc_real_outer = outer_encoder_discriminator.evaluate(mnist_images,
                                                                           y_ones)
    d_loss_outer, d_acc_outer = 0.5 * \
        np.add(loss_fake_outer, loss_real_outer), 0.5 * \
        np.add(acc_fake_outer, acc_real_outer)

    encoder_discriminator_loss_outer.append(d_loss_outer)
    encoder_discriminator_accuracy_outer.append(d_acc_outer)

    g_loss_outer, g_acc_outer = outer_encoder_gan.evaluate(
        original_images_scaled, y_ones)

    encoder_generator_loss_outer.append(g_loss_outer)
    encoder_generator_accuracy_outer.append(g_acc_outer)

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

    # Evaluate
    loss_fake_inner, acc_fake_inner = inner_encoder_discriminator.evaluate(inner_encoded_images_scaled,
                                                                           y_zeros)
    loss_real_inner, acc_real_inner = inner_encoder_discriminator.evaluate(mnist_images,
                                                                           y_ones)
    d_loss_inner, d_acc_inner = 0.5 * \
        np.add(loss_fake_inner, loss_real_inner), 0.5 * \
        np.add(acc_fake_inner, acc_real_inner)

    encoder_discriminator_loss_inner.append(d_loss_inner)
    encoder_discriminator_accuracy_inner.append(d_acc_inner)

    g_loss_inner, g_acc_inner = inner_encoder_gan.evaluate(
        mnist_images, y_ones)

    encoder_generator_loss_inner.append(g_loss_inner)
    encoder_generator_accuracy_inner.append(g_acc_inner)

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

    # Evaluate
    loss_inner, accuracy_inner = inner_decoder_gan.evaluate(
        mnist_images, mnist_images)

    decoder_loss_inner.append(loss_inner)
    decoder_accuracy_inner.append(accuracy_inner)

    """ Decoder - Outer layer """

    # Decode images
    outer_decoded_images_scaled = outer_decoder_generator.predict(
        inner_decoded_images_scaled)

    # Display decoded images
    outer_decoded_name = '{} - 5 - Outer - Decoded'.format(current_round + 1)
    outer_decoded_images = (outer_decoded_images_scaled + 1) / 2 * 255

    labels_probs = classifier.predict(outer_decoded_images)
    decoded_labels = []
    for probs in labels_probs:
        label = np.argmax(probs)
        decoded_labels.append(label)

    outer_decoded_images = outer_decoded_images[:, :, :, 0]
    image_displayer.display_samples(name=outer_decoded_name,
                                    samples=outer_decoded_images,
                                    should_display_directly=should_display_directly,
                                    should_save_to_file=should_save_to_file,
                                    labels=decoded_labels)

    # Evaluate
    loss_outer, accuracy_outer = outer_decoder_gan.evaluate(
        original_images_scaled, original_images_scaled)

    decoder_loss_outer.append(loss_outer)
    decoder_accuracy_outer.append(accuracy_outer)

    loss_class, acc_class = classifier.evaluate(outer_decoded_images,
                                                original_labels)
    class_loss.append(loss_class)
    class_accuracy.append(acc_class)
    print('Decoded classification loss: {}, accuracy: {}'.format(
        loss_class, acc_class))

    # Calculate recall and precision and f1 score
    confusion = confusion_matrix(original_labels,
                                 decoded_labels)
    confusion_name = 'Confusion Matrix - {}'.format(current_round + 1)
    confusion_displayer.display_samples(name=confusion_name,
                                        samples=confusion,
                                        should_display_directly=should_display_directly,
                                        should_save_to_file=should_save_to_file)

    classification = classification_report(original_labels,
                                           decoded_labels)
    report = {
        'classification': classification,
        'loss_class_original': loss_class_original,
        'acc_class_original': acc_class_original,

        'encoder_discriminator_loss_outer': d_loss_outer,
        'encoder_discriminator_accuracy_outer': d_acc_outer,
        'encoder_generator_loss_outer': g_loss_outer,
        'encoder_generator_accuracy_outer': g_acc_outer,

        'encoder_discriminator_loss_inner': d_loss_inner,
        'encoder_discriminator_accuracy_inner': d_acc_inner,
        'encoder_generator_loss_inner': g_loss_inner,
        'encoder_generator_accuracy_inner': g_acc_inner,

        'decoder_loss_inner': loss_inner,
        'decoder_accuracy_inner': accuracy_inner,

        'decoder_loss_outer': loss_outer,
        'decoder_accuracy_outer': accuracy_outer,

        'loss_class': loss_class,
        'acc_class': acc_class
    }

diagram_displayer.display_samples(name='Outer Encoder Discriminator Loss',
                                  samples=encoder_discriminator_loss_outer,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Outer Encoder Discriminator Accuracy',
                                  samples=encoder_discriminator_accuracy_outer,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Inner Encoder Discriminator Loss',
                                  samples=encoder_discriminator_loss_inner,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Inner Encoder Discriminator Accuracy',
                                  samples=encoder_discriminator_accuracy_inner,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Outer Encoder Generator Loss',
                                  samples=encoder_generator_loss_outer,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Outer Encoder Generator Accuracy',
                                  samples=encoder_generator_accuracy_outer,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Inner Encoder Generator Loss',
                                  samples=encoder_generator_loss_inner,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Inner Encoder Generator Accuracy',
                                  samples=encoder_generator_accuracy_inner,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Outer Decoder Loss',
                                  samples=decoder_loss_outer,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Outer Decoder Accuracy',
                                  samples=decoder_accuracy_outer,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Inner Decoder Loss',
                                  samples=decoder_loss_inner,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

diagram_displayer.display_samples(name='Inner Decoder Accuracy',
                                  samples=decoder_accuracy_inner,
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