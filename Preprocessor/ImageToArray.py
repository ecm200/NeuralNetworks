import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat
        
    def preprocess(self, image):
        # apply Keras utlity function correclty rearranges image dims
        return img_to_array(image, data_format=self.dataFormat)