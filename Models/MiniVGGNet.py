import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


class MiniVGGNet:

    @staticmethod
    def build(width, height, depth, classes):
        # Init model with "channels last"
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        # If using "channels first", update input shape and channels dimension
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, width, height)
            chanDim = 1
            
        # First CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
                  
        # First (only) set of Dense layers.
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        return model