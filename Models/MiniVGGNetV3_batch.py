import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input, Lambda
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import numpy as np

class MiniVGGNetV3_batch:

    @staticmethod
    def build(width, height, depth, classes,
              batch=True, initConvUnits=32, initConvOp=(3, 3),
              initMaxPoolSize=(2, 2), initDropout=0.25,
              nDeepLayers=1, deepConvUnits=[64], deepConvOp=[(3, 3)],
              deepMaxPoolSize=[(2, 2)], deepDropout=[0.25],
              fcUnits=512, fcDropout=0.5, imgProcFunc=None):
        
        # Init model with "channels last"
        #model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        # If using "channels first", update input shape and channels dimension
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, width, height)
            chanDim = 1

        inputs = Input(shape=inputShape)
        
        ## LAYERS START ##
        # Pre-process image files of the current batch
        if imgProcFunc is not None:
            x = Lambda(imgProcFunc)(inputs)
        else:
            x = inputs

        # First CONV => RELU => CONV => RELU => POOL layer set
        x = Conv2D(initConvUnits, initConvOp, padding='same')(x)
        x = Activation('relu')(x)
        if batch:
            x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(initConvUnits, initConvOp, padding='same')(x)
        x = Activation('relu')(x)
        if batch:
            x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=initMaxPoolSize)(x)
        x = Dropout(initDropout)(x)
        
        # Second CONV => RELU => CONV => RELU => POOL layer set
        for deepLayer in np.arange(0,nDeepLayers, 1):
            x = Conv2D(deepConvUnits[deepLayer], deepConvOp[deepLayer], padding='same')(x)
            x = Activation('relu')(x)
            if batch:
                x = BatchNormalization(axis=chanDim)(x)
            x = Conv2D(deepConvUnits[deepLayer], deepConvOp[deepLayer], padding='same')(x)
            x = Activation('relu')(x)
            if batch:
                x = BatchNormalization(axis=chanDim)(x)
            x = MaxPooling2D(pool_size=deepMaxPoolSize[deepLayer])(x)
            x = Dropout(deepDropout[deepLayer])(x)
                  
        # First (only) set of Dense layers.
        x = Flatten()(x)
        x = Dense(fcUnits)(x)
        x = Activation('relu')(x)
        if batch:
            x = BatchNormalization()(x)
        x = Dropout(fcDropout)(x)
        
        # Softmax classifier
        x = Dense(classes)(x)
        x = Activation('softmax')(x)

        # Create the model
        model = Model(inputs, x, name='miniVGGnet')
        
        return model
