from tensorflow.keras.datasets import fashion_mnist
from bpp_calculator import BppCalculator
import fakesafe_constants as constants
import numpy as np
import os
import re
from PIL import Image

(fashion_images, _), _ = fashion_mnist.load_data()

face_images = []
for face_file in os.scandir(constants.FACE_IMAGE_DATASET_PATH):

    face_file_name = face_file.name
    face_file_path = constants.FACE_IMAGE_DATASET_PATH + '/' + face_file_name
    face_image = Image.open(face_file_path)
    resized_image = face_image.resize((28, 28))
    image_array = np.asarray(resized_image)
    face_images.append(image_array)

face_images = np.asarray(face_images)

calculator = BppCalculator()
bpp = calculator.calculate_bpp(face_images, fashion_images)

print('Face -> Fashion -> Face BPP: {}'.format(bpp))
