from import_resources import *

# Create the process_image function
def process_image(image):
    image_size = 224
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image,(image_size,image_size))
    image = tf.cast(image,tf.float32)
    image /= 255
    return image.numpy()