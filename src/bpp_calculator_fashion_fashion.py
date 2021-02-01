from tensorflow.keras.datasets import fashion_mnist
from bpp_calculator import BppCalculator

(fashion_images, _), _ = fashion_mnist.load_data()

calculator = BppCalculator()
bpp = calculator.calculate_bpp(fashion_images, fashion_images)

print('Fashion -> Fashion -> Fashion BPP: {}'.format(bpp))
