from tensorflow.keras.models import load_model

from displayers import SampleImageDisplayer, SampleDiagramDisplayer, SampleReportDisplayer
from imblearn.over_sampling import RandomOverSampler

import numpy as np
import constants

import sys
import os
from PIL import Image

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
pain_level_labels = []

# Load pain level data
pain_level_dict = {}
for subdir, dirs, files in os.walk('../data/pain'):
    for file_name in files:
        if file_name.endswith('txt'):
            file_path = subdir + os.sep + file_name
            with open(file_path, 'r') as f:
                pain_level = f.read().split('\n')[0]
                pain_level = float(pain_level)
            file_name = file_name.split('_')[0]
            pain_level_dict[file_name] = pain_level

# Load face images
for face_file in os.scandir(constants.FACE_IMAGE_DATASET_PATH):

    face_file_name = face_file.name

    face_name = face_file_name.split('.png')[0]
    pain_level = pain_level_dict.get(face_name)

    if pain_level is not None:
        pain_level_labels.append(pain_level)
        face_file_path = constants.FACE_IMAGE_DATASET_PATH + '/' + face_file_name
        face_image = Image.open(face_file_path)
        grey_image = face_image.convert('L')
        resized_image = grey_image.resize((28, 28))
        image_array = np.asarray(resized_image)
        face_images.append(image_array)
    else:
        print('Not found: {}'.format(face_file_name))

# Over sampling the minority data
oversample = RandomOverSampler(sampling_strategy='minority')

face_images = np.asarray(face_images)
pain_level_labels = np.asarray(pain_level_labels)

nsamples, nx, ny = face_images.shape
face_images = face_images.reshape((nsamples, nx * ny))
face_images, pain_level_labels = oversample.fit_resample(
    face_images, pain_level_labels)

print('Face images shape: {}'.format(face_images.shape))
print('Pain level shape: {}'.format(pain_level_labels.shape))

nsamples = face_images.shape[0]
face_images = face_images.reshape((nsamples, nx, ny))

print('Face images shape: {}'.format(face_images.shape))
print('Pain level shape: {}'.format(pain_level_labels.shape))

# Select sample of images
original_indexes = np.random.randint(0,
                                     face_images.shape[0],
                                     constants.DISPLAY_ROW * constants.DISPLAY_COLUMN)
original_images = face_images[original_indexes]
original_labels = pain_level_labels[original_indexes]

"""
Load models
"""

# Load classifier
try:
    classifier = load_model('model/classifier_face_pain.h5')
except ImportError:
    print('Unable to load classifier. Please run classifier script first')
    sys.exit()

# Load encoder and decoder
try:
    encoder_generator = load_model(
        'model/one_layer_face_mnist_encoder_generator.h5')
    decoder_generator = load_model(
        'model/one_layer_face_mnist_decoder_generator.h5')
except ImportError:
    print('Unable to load encoder and decoder. Please fun "one_layer_mnist" script first')
    sys.exit()

""" Face -> MNIST -> Face """

image_displayer_gray = SampleImageDisplayer(row=constants.DISPLAY_ROW,
                                            column=constants.DISPLAY_COLUMN,
                                            cmap='gray')

# Display original images
original_name = 'Original - Face -> MNIST -> Face'
image_displayer_gray.display_samples(name=original_name,
                                     samples=original_images,
                                     should_display_directly=should_display_directly,
                                     should_save_to_file=should_save_to_file,
                                     labels=original_labels)

# Encode images
original_images_scaled = (original_images / 255.0) * 2 - 1
encoded_images_scaled = encoder_generator.predict(original_images_scaled)

# Display encoded images
encoded_name = 'Encoded - Face -> MNIST -> Face'
encoded_images = (encoded_images_scaled + 1) / 2 * 255
encoded_images = encoded_images[:, :, :, 0]
image_displayer_gray.display_samples(name=encoded_name,
                                     samples=encoded_images,
                                     should_display_directly=should_display_directly,
                                     should_save_to_file=should_save_to_file)

# Decode images
decoded_images_scaled = decoder_generator.predict(encoded_images_scaled)

# Display decoded images
decoded_name = 'Decoded - Face -> MNIST -> Face'
decoded_images = (decoded_images_scaled + 1) / 2 * 255
decoded_images = decoded_images[:, :, :, 0]

decoded_labels = classifier.predict(decoded_images)
decoded_labels = [round(x[0]) for x in decoded_labels]

print(original_labels)
print(decoded_labels)

image_displayer_gray.display_samples(name=decoded_name,
                                     samples=decoded_images,
                                     should_display_directly=should_display_directly,
                                     should_save_to_file=should_save_to_file,
                                     labels=decoded_labels)

# Evaluate
loss_original, accuracy_original = classifier.evaluate(original_images,
                                                       original_labels)
loss_decoded, accuracy_decoded = classifier.evaluate(decoded_images,
                                                     original_labels)

print('Loss original: {}, accuracy original: {}'.format(
    loss_original, accuracy_original))
print('Loss decoded: {}, accuracy decoded: {}'.format(
    loss_decoded, accuracy_decoded))
