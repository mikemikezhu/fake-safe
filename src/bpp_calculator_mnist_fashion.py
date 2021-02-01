from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from bpp_calculator import BppCalculator

(mnist_images, _), _ = mnist.load_data()
(fashion_images, _), _ = fashion_mnist.load_data()

calculator = BppCalculator()
bpp = calculator.calculate_bpp(mnist_images, fashion_images)

print('MNIST -> Fashion -> MNIST BPP: {}'.format(bpp))
