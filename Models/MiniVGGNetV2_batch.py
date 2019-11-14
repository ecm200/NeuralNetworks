import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input, Lambda
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import numpy as np

class MiniVGGNetV2_batch:

    @staticmethod
    def build(imgProcFunc, width, height, depth, classes,
              batch=True, initConvUnits=32, initConvOp=(3, 3),
              initMaxPoolSize=(2, 2), initDropout=0.25,
              nDeepLayers=1, deepConvUnits=[64], deepConvOp=[(3, 3)],
              deepMaxPoolSize=[(2, 2)], deepDropout=[0.25],
              fcUnits=512, fcDropout=0.5):
        
        # Init model with "channels last"
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        # If using "channels first", update input shape and channels dimension
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, width, height)
            chanDim = 1
        
        ## LAYERS START ##
        # Pre-process image files of the current batch
        model.add(Lambda(imgProcFunc, input_shape=inputShape))
        
        # First CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(initConvUnits, initConvOp, padding='same'))
        model.add(Activation('relu'))
        if batch:
            model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(initConvUnits, initConvOp, padding='same'))
        model.add(Activation('relu'))
        if batch:
            model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=initMaxPoolSize))
        model.add(Dropout(initDropout))
        
        # Second CONV => RELU => CONV => RELU => POOL layer set
        for deepLayer in np.arange(0,nDeepLayers, 1):
            model.add(Conv2D(deepConvUnits[deepLayer], deepConvOp[deepLayer], padding='same'))
            model.add(Activation('relu'))
            if batch:
                model.add(BatchNormalization(axis=chanDim))
            model.add(Conv2D(deepConvUnits[deepLayer], deepConvOp[deepLayer], padding='same'))
            model.add(Activation('relu'))
            if batch:
                model.add(BatchNormalization(axis=chanDim))
            model.add(MaxPooling2D(pool_size=deepMaxPoolSize[deepLayer]))
            model.add(Dropout(deepDropout[deepLayer]))
                  
        # First (only) set of Dense layers.
        model.add(Flatten())
        model.add(Dense(fcUnits))
        model.add(Activation('relu'))
        if batch:
            model.add(BatchNormalization())
        model.add(Dropout(fcDropout))
        
        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        return model
