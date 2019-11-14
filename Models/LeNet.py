import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


class LeNet:
    
    @staticmethod
    def build(width, height, depth, classes):
        # Init model
        model = Sequential()
        inputShape = (height, width, depth)
        
        # If using "channels first" update input shape
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            
        # First set of layers CONV => RELU => POOL
        model.add(Conv2D(20, (5,5), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # Second set set of layers CONV => RELU => POOL
        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # First and only set of DENSE FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        
        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        return model
