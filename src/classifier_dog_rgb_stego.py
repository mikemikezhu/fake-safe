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

# The classifier will distinguish between ImageNet dog images and face images
# For example, for one layer FakeSafe: Fashion -> Dog -> Fashion, 
# we can use this classifier to check whether the stego images are dog images

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
print('Classifier Dog RGB Stego')
print('************************')

# Load data

model_name = 'classifier_dog_rgb_stego'

dog_images = []
dog_counter = 0
total_dog_images = 3000

for dog_file in os.scandir(constants.DOG_IMAGE_DATASET_PATH):
    if dog_counter < total_dog_images:
        dog_file_name = dog_file.name
        if dog_file_name.endswith('.jpg'):
            dog_file_path = constants.DOG_IMAGE_DATASET_PATH + '/' + dog_file_name
            dog_image = Image.open(dog_file_path)
            image_array = np.asarray(dog_image)
            dog_images.append(image_array)
            dog_counter += 1
    else:
        break

dog_images = np.asarray(dog_images)
dog_labels = np.ones((total_dog_images,))

face_images = []
face_counter = 0
total_face_images = 3000

for face_file in os.scandir(constants.FACE_IMAGE_DATASET_PATH):
    if face_counter < total_face_images:
        face_file_name = face_file.name
        face_file_path = constants.FACE_IMAGE_DATASET_PATH + '/' + face_file_name
        face_image = Image.open(face_file_path)
        resized_image = face_image.resize((256, 256))
        image_array = np.asarray(resized_image)
        face_images.append(image_array)
        face_counter += 1
    else:
        break

face_images = np.asarray(face_images)
face_labels = np.zeros((total_face_images,))

stego_images = np.concatenate((dog_images, face_images))
stego_labels = np.concatenate((dog_labels, face_labels))

x_train, x_test, y_train, y_test = train_test_split(stego_images,
                                                    stego_labels,
                                                    test_size=0.15)
print('x_train: {}'.format(x_train.shape))
print('x_test: {}'.format(x_test.shape))
print('y_train: {}'.format(y_train.shape))
print('y_test: {}'.format(y_test.shape))

# Create classifier
classifier_creator = ClassifierModelCreator(constants.IMAGE_NET_RGB_INPUT_SHAPE,
                                            2,
                                            model_name,
                                            activation='sigmoid')
classifier = classifier_creator.create_model()

# Train classifier
trainer = ClassifierTrainer(classifier,
                            constants.TRAINING_EPOCHS,
                            constants.BATCH_SIZE)
accuracy, val_accuracy = trainer.train(x_train, y_train, x_test, y_test)

# Plot history
diagram_displayer = SampleDiagramDisplayer()
diagram_displayer.display_samples(name='Classifier Stego RGB Accuracy',
                                  samples=accuracy,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)
diagram_displayer.display_samples(name='Classifier Stego RGB Validation Accuracy',
                                  samples=val_accuracy,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

# Save classifier model for future use
model_path = 'model/{}.h5'.format(model_name)
classifier.save(model_path)
