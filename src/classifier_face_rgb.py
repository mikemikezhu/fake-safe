from sklearn.model_selection import train_test_split

from classifier_models import ClassifierModelCreator
from trainers import ClassifierTrainer
from displayers import SampleDiagramDisplayer

import fakesafe_constants as constants
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
print('Classifier Face RGB')
print('************************')

# Load data
face_images = []
face_labels = []
model_name = 'classifier_face_rgb'

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

x_train, x_test, y_train, y_test = train_test_split(face_images,
                                                    face_labels,
                                                    test_size=0.15)
print('x_train: {}'.format(x_train.shape))
print('x_test: {}'.format(x_test.shape))
print('y_train: {}'.format(y_train.shape))
print('y_test: {}'.format(y_test.shape))

# Create classifier
classifier_creator = ClassifierModelCreator(constants.DEFAULT_RGB_INPUT_SHAPE,
                                            10,
                                            model_name)
classifier = classifier_creator.create_model()

# Train classifier
trainer = ClassifierTrainer(classifier,
                            constants.TRAINING_EPOCHS,
                            constants.BATCH_SIZE)
accuracy, val_accuracy = trainer.train(x_train, y_train, x_test, y_test)

# Plot history
diagram_displayer = SampleDiagramDisplayer()
diagram_displayer.display_samples(name='Classifier Face RGB Accuracy',
                                  samples=accuracy,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)
diagram_displayer.display_samples(name='Classifier Face RGB Validation Accuracy',
                                  samples=val_accuracy,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

# Save classifier model for future use
model_path = 'model/{}.h5'.format(model_name)
classifier.save(model_path)
