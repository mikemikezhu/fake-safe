from tensorflow.keras.models import load_model

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

from generator_models import GeneratorModelCreator
from discriminator_models import DiscriminatorModelCreator
from gan_models import EncoderGanModelCreator, DecoderGanModelCreator

from trainers import EncoderTrainer, DecoderTrainer
from displayers import SampleImageDisplayer, SampleDiagramDisplayer, SampleConfusionMatrixDisplayer, SampleReportDisplayer

from numpy import ones
from numpy import zeros
import numpy as np
import fakesafe_constants as constants

import sys
import os
import re
from PIL import Image

"""
One layer encoding-decoding:
Dog -> (Encode) -> Dog -> (Decode) -> Dog
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

dog_images = []
# dog_labels = []

for dog_file in os.scandir(constants.DOG_IMAGE_DATASET_PATH):
    dog_file_name = dog_file.name
    if dog_file_name.endswith('.jpg'):
        # dog_label = re.findall('\d+', dog_file_name)[0]
        # dog_labels.append(dog_label)

        dog_file_path = constants.DOG_IMAGE_DATASET_PATH + '/' + dog_file_name
        dog_image = Image.open(dog_file_path)
        # resized_image = dog_image.resize((28, 28))
        image_array = np.asarray(dog_image)
        dog_images.append(image_array)

dog_images = np.asarray(dog_images)
# dog_labels = np.asarray(dog_labels)

# unique_labels = np.unique(dog_labels)
# labels_to_index = {}
# for index in range(unique_labels.shape[0]):
#     label = unique_labels[index]
#     labels_to_index[label] = index
# print(labels_to_index)

# dog_labels = [labels_to_index[label] for label in dog_labels]
# dog_labels = np.asarray(dog_labels)

print('dog images shape: {}'.format(dog_images.shape))
# print('dog labels shape: {}'.format(dog_labels.shape))

# Load data
# dog_images_train, dog_images_test, dog_labels_train, dog_labels_test = train_test_split(dog_images,
#                                                                                         dog_labels,
#                                                                                         test_size=0.15)

dog_images_train, dog_images_test = train_test_split(dog_images, test_size=0.15)

# Rescale -1 to 1
dog_images_train_scaled = (dog_images_train / 255.0) * 2 - 1
dog_images_test_scaled = (dog_images_test / 255.0) * 2 - 1

"""
Create models
"""

# Classifier
# try:
#     classifier = load_model('model/classifier_dog_rgb.h5')
# except ImportError:
#     print('Unable to load classifier. Please run classifier script first')
#     sys.exit()

# Encoder

# Create encoder generator
encoder_generator_creator = GeneratorModelCreator(constants.IMAGE_NET_RGB_INPUT_SHAPE,
                                                  constants.IMAGE_NET_RGB_OUTPUT_SHAPE,
                                                  from_image=True,
                                                  to_image=True,
                                                  activation='tanh')
encoder_generator = encoder_generator_creator.create_model()

# Create encoder discriminator
encoder_discriminator_creator = DiscriminatorModelCreator(
    constants.IMAGE_NET_RGB_INPUT_SHAPE)
encoder_discriminator = encoder_discriminator_creator.create_model()

# Create GAN model to combine encoder generator and discriminator
encoder_gan_creator = EncoderGanModelCreator(encoder_generator,
                                             encoder_discriminator)
encoder_gan = encoder_gan_creator.create_model()

# Decoder

# Create decoder generator
decoder_generator_creator = GeneratorModelCreator(constants.IMAGE_NET_RGB_INPUT_SHAPE,
                                                  constants.IMAGE_NET_RGB_OUTPUT_SHAPE,
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

image_displayer_rgb = SampleImageDisplayer(row=constants.IMAGE_NET_DISPLAY_ROW,
                                           column=constants.IMAGE_NET_DISPLAY_COLUMN)

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

y_zeros = zeros((constants.IMAGE_NET_DISPLAY_ROW * constants.IMAGE_NET_DISPLAY_COLUMN, 1))
y_ones = ones((constants.IMAGE_NET_DISPLAY_ROW * constants.IMAGE_NET_DISPLAY_COLUMN, 1))

for current_round in range(constants.TOTAL_TRAINING_ROUND):

    print('************************')
    print('Round: {}'.format(current_round + 1))
    print('************************')

    # Train encoder
    encoder_trainer.train(input_data=dog_images_train_scaled,
                          exp_output_data=dog_images_train_scaled)
    # Train decoder
    decoder_trainer.train(input_data=dog_images_train_scaled)

    # Select sample of images
    sample_indexes = np.random.randint(0,
                                       dog_images_test.shape[0],
                                       constants.IMAGE_NET_DISPLAY_ROW * constants.IMAGE_NET_DISPLAY_COLUMN)
    sample_images = dog_images_test[sample_indexes]
    # sample_labels = dog_labels_test[sample_indexes]

    # Display original images
    original_name = 'Original - {}'.format(current_round + 1)
    image_displayer_rgb.display_samples(name=original_name,
                                        samples=sample_images,
                                        should_display_directly=should_display_directly,
                                        should_save_to_file=should_save_to_file)
                                        # labels=sample_labels)

    # Evaluate images with labels
    # loss_class_original, acc_class_original = classifier.evaluate(sample_images,
    #                                                               sample_labels)
    # print('Original classification loss: {}, accuracy: {}'.format(
    #     loss_class_original, acc_class_original))

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

    # labels_probs = classifier.predict(decoded_sample_images)
    # decoded_sample_labels = []
    # for probs in labels_probs:
    #     label = np.argmax(probs)
    #     decoded_sample_labels.append(label)

    image_displayer_rgb.display_samples(name=decoded_name,
                                        samples=decoded_sample_images,
                                        should_display_directly=should_display_directly,
                                        should_save_to_file=should_save_to_file)
                                        # labels=decoded_sample_labels)

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

    # Compare PSNR and SSIM
    total_sample_images = constants.IMAGE_NET_DISPLAY_ROW * constants.IMAGE_NET_DISPLAY_COLUMN
    total_ssim = 0
    total_psnr = 0
    
    for index in range(total_sample_images):
        
        sample_image = sample_images[index]
        decoded_sample_image = decoded_sample_images[index]
        
        ssim = structural_similarity(sample_image, decoded_sample_image, multichannel=True)
        psnr = peak_signal_noise_ratio(sample_image, decoded_sample_image)
        
        total_ssim += ssim
        total_psnr += psnr
    
    avg_ssim = total_ssim / total_sample_images
    avg_psnr = total_psnr / total_sample_images

    print('SSIM: {}'.format(avg_ssim))
    print('PSNR: {}'.format(avg_psnr))

    # Evaluate images with labels
    # loss_class, acc_class = classifier.evaluate(decoded_sample_images,
    #                                             sample_labels)
    # class_loss.append(loss_class)
    # class_accuracy.append(acc_class)
    # print('Decoded classification loss: {}, accuracy: {}'.format(
    #     loss_class, acc_class))

    # Calculate recall and precision and f1 score
    # confusion = confusion_matrix(sample_labels,
    #                              decoded_sample_labels)
    # confusion_name = 'Confusion Matrix - {}'.format(current_round + 1)
    # confusion_displayer.display_samples(name=confusion_name,
    #                                     samples=confusion,
    #                                     should_display_directly=should_display_directly,
    #                                     should_save_to_file=should_save_to_file)

    # classification = classification_report(sample_labels,
    #                                        decoded_sample_labels)
    # report = {
    #     'classification': classification,
    #     'loss_class_original': loss_class_original,
    #     'acc_class_original': acc_class_original,
    #     'encoder_discriminator_loss': d_loss,
    #     'encoder_discriminator_accuracy': d_acc,
    #     'encoder_generator_loss': g_loss,
    #     'encoder_generator_accuracy': g_acc,
    #     'decoder_loss': loss,
    #     'decoder_accuracy': accuracy,
    #     'loss_class': loss_class,
    #     'acc_class': acc_class
    # }

    report = {
        'encoder_discriminator_loss': d_loss,
        'encoder_discriminator_accuracy': d_acc,
        'encoder_generator_loss': g_loss,
        'encoder_generator_accuracy': g_acc,
        'decoder_loss': loss,
        'decoder_accuracy': accuracy,
        'ssim': avg_ssim,
        'psnr': avg_psnr
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

encoder_generator.save('model/one_layer_dog_dog_rgb_encoder_generator.h5')
decoder_generator.save('model/one_layer_dog_dog_rgb_decoder_generator.h5')
