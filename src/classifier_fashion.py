from tensorflow.keras.datasets import fashion_mnist

from classifier_models import ClassifierModelCreator
from trainers import ClassifierTrainer
from displayers import SampleDiagramDisplayer

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
print('Classifier Fashion')
print('************************')

# Load data
model_name = 'classifier_fashion'
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print('x_train: {}'.format(x_train.shape))
print('x_test: {}'.format(x_test.shape))
print('y_train: {}'.format(y_train.shape))
print('y_test: {}'.format(y_test.shape))

# Create classifier
classifier_creator = ClassifierModelCreator(constants.INPUT_SHAPE,
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
diagram_displayer.display_samples(name='Classifier Fashion Accuracy',
                                  samples=accuracy,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)
diagram_displayer.display_samples(name='Classifier Fashion Validation Accuracy',
                                  samples=val_accuracy,
                                  should_display_directly=should_display_directly,
                                  should_save_to_file=should_save_to_file)

# Save classifier model for future use
model_path = 'model/{}.h5'.format(model_name)
classifier.save(model_path)
