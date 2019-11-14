import tensorflow
from tensorflow.keras.layers import BatchNormalization, Conv2D, AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Input, add
from tensorflow.keras.model import Model
from tensorflow.keras.regularizers import l2
from keras import backend as K

class ResNet:

    # Residual module with pre-activation and bottleneck architecture
    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False, reg=0.0001, 
                            bnEps=2e-5, bnMom=0.9):

        # Shortcut branch of the ResNet module should be initlialize as input (identity) data
        shortcut = data

        # first block of module is 1x1 CONVs
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation('relu')(bn1)
        conv1 = Conv2D(int(K*0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        # second block of module are 3x3 CONVs
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K*0.25), (3, 3), strides=stride, padding='same', use_bias=False,
                        kernel_regularizer=l2(reg))(act2)
        
        # third block of module another set of 1x1 CONVs
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        # If reducing the spatial size, apply a CONV layer to shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False,
                                kernel_regularizer=l2(reg))(act1)
        
        # Add the shortcut and CONV3 output together
        x = add([conv3, shortcut])
        
        return x

    
    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset='cifar'):

        # Init the input shape to "channels last" and the channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1
        
        # If using "channels first" update input shape and channel dimensions
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            chanDim = 1

        inputs = Inpute(shape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(inputs)

        # TODO Make this a function pass rather than a IF statement so that the dataset defined 
        # layers can be passed in a functional argument.
        if dataset == "cifar":
            # apply single CONV layer
            x = Conv2D(filters[0], (3, 3), use_bias=False, padding='same',
                        kernel_regularizer=l2(reg))(x)
        
        # Loop over the number of stages
        for i in range(0, len(stages)):

            # Init the stride, then apply residual module used to reduce spatial size of 
            # input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride, chanDim, red=True,
                                        bnEps=bnEps, bnMom=bnMom)
            
            # Loop over number of layers in the stage
            for j in range(0, stages[i] - 1):
                # Apply ResNet module
                x = ResNet.residual_module(x, filters[i + 1], (1, 1), chanDim,
                                            bnEps=bnEps, bnMom=bnMom)
                
        # To avoid using Dense layers, instead apply average pooling to reduce volume size 
        # to 1 x 1 x classes.
        # Apply BN => ACT => POOL
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        # Softmax classifier
        x = Flatten(x)
        x = Dense(classes, kernel_regularizerl2(reg))(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name="resnet")

        return model