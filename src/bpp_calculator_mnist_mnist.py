from tensorflow.keras.datasets import mnist
from bpp_calculator import BppCalculator

(mnist_images, _), _ = mnist.load_data()

calculator = BppCalculator()
bpp = calculator.calculate_bpp(mnist_images, mnist_images)

print('Mnist -> Mnist -> Mnist BPP: {}'.format(bpp))
