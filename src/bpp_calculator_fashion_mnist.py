from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from bpp_calculator import BppCalculator

(mnist_images, _), _ = mnist.load_data()
(fashion_images, _), _ = fashion_mnist.load_data()

calculator = BppCalculator()
bpp = calculator.calculate_bpp(fashion_images, mnist_images)

print('Fashion -> MNIST -> Fashion BPP: {}'.format(bpp))
