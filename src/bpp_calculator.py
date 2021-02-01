class BppCalculator:

    def calculate_bpp(self, secret_images, container_images):

        print('Secret images shape: {}'.format(secret_images.shape))
        print('Container images shape: {}'.format(container_images.shape))

        # Calculate secret image size in bits
        total_secret_images_bytes = secret_images.nbytes
        total_secret_images_bits = total_secret_images_bytes * 8
        total_secret_images = secret_images.shape[0]
        secret_image_size = total_secret_images_bits / total_secret_images

        print('Secret image size: {}'.format(secret_image_size))

        # Calculate container image pixels
        container_image_width = container_images.shape[1]
        container_image_height = container_images.shape[2]
        container_image_pixels = container_image_width * container_image_height

        return secret_image_size / container_image_pixels
