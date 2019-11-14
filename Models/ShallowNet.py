import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


class ShallowNet:
    
    @staticmethod
    def build(width, height, depth, classes):
        # Init the model along with input shape, to be channels last
        model = Sequential()
        inputShape = (height, width, depth)
        
        # if we are using "channel first", update the input shape
        if K.image_data_format() == 'channel_first':
            inputShape = (depth, height, wdith)
            
        # Define first (and only CONV ==> RELU layer)
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=inputShape))
        model.add(Activation('relu'))
        
        # Softmas classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        # return the constructed network architecture
        return model

