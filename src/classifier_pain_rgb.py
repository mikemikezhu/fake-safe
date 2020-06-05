from sklearn.model_selection import train_test_split

from classifier_models import ClassifierModelCreator
from trainers import ClassifierTrainer
from displayers import SampleDiagramDisplayer

from imblearn.over_sampling import RandomOverSampler

import constants
import os
import sys
import re
import numpy as np
from PIL import Image

# Parse arguments
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

print('************************')
print('Classifier Face Pain RGB')
print('************************')

# Load data

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

face_images = []
pain_level_labels = []
model_name = 'classifier_face_pain_rgb'

for face_file in os.scandir(constants.FACE_IMAGE_DATASET_PATH):

    face_file_name = face_file.name

    face_name = face_file_name.split('.png')[0]
    pain_level = pain_level_dict.get(face_name)

    if pain_level is not None:
        pain_level_labels.append(pain_level)
        face_file_path = constants.FACE_IMAGE_DATASET_PATH + '/' + face_file_name
        face_image = Image.open(face_file_path)
        resized_image = face_image.resize((28, 28))
        image_array = np.asarray(resized_image)
        face_images.append(image_array)
    else:
        print('Not found: {}'.format(face_file_name))

# Over sampling the minority data
oversample = RandomOverSampler(sampling_strategy='minority')

face_images = np.asarray(face_images)
pain_level_labels = np.asarray(pain_level_labels)

nsamples, nx, ny, nz = face_images.shape
face_images = face_images.reshape((nsamples, nx*ny*nz))
face_images, pain_level_labels = oversample.fit_resample(
    face_images, pain_level_labels)

print('Face images shape: {}'.format(face_images.shape))
print('Pain level shape: {}'.format(pain_level_labels.shape))

nsamples = face_images.shape[0]
face_images = face_images.reshape((nsamples, nx, ny, nz))

print('Face images shape: {}'.format(face_images.shape))
print('Pain level shape: {}'.format(pain_level_labels.shape))

x_train, x_test, y_train, y_test = train_test_split(face_images,
                                                    pain_level_labels,
                                                    test_size=0.15)
print('x_train: {}'.format(x_train.shape))
print('x_test: {}'.format(x_test.shape))
print('y_train: {}'.format(y_train.shape))
print('y_test: {}'.format(y_test.shape))

# Create classifier
classifier_creator = ClassifierModelCreator(constants.RGB_INPUT_SHAPE,
                                            1,
                                            model_name,
                                            activation='linear',
                                            loss='mse')
classifier = classifier_creator.create_model()

# Train classifier
trainer = ClassifierTrainer(classifier,
                            100,
                            constants.BATCH_SIZE,
                            should_early_stopping=False)
accuracy, val_accuracy = trainer.train(x_train, y_train, x_test, y_test)

# Plot history
diagram_displayer = SampleDiagramDisplayer()
diagram_displayer.display_samples(name='Classifier Face Pain Accuracy',
                                  samples=accuracy,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)
diagram_displayer.display_samples(name='Classifier Face Pain Validation Accuracy',
                                  samples=val_accuracy,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

# Save classifier model for future use
model_path = 'model/{}.h5'.format(model_name)
classifier.save(model_path)

# Display sample labels
output = classifier.predict(x_test[:10])
output = [round(x[0]) for x in output]
print('Output: {}'.format(output))
print('Labels: {}'.format(y_test[:10]))
