import tensorflow as tf

def image_process(x):
    # Preprocess the image files from 3 colour RGB to grayscale. Changes depth of input array to 1.
    hsv = tf.image.rgb_to_hsv(x)
    gray = tf.image.rgb_to_grayscale(x)
    rez = tf.concat([hsv, gray], axis=-1)
    return rez

def image_load_and_process(imagePath):
    # Load the image from file and decode and preprocess
    img = tf.io.read_file(imagePath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = image_process(img)
    return img