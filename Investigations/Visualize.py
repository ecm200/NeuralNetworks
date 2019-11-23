import tensorflow
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image

import numpy as np

import matplotlib.pyplot as plt

import cv2

def createActivationModel(model, img_tensor, maxLayer):
    ##
    #
    # model         OBJECT      Compiled Keras model object.
    #
    # imag_tensor   4D ARRAY    Numpy 4D array of size 1 x NX x NY x 3, where NX and NY 
    #                           are image pixel size in X and Y dims respectively.
    #
    # maxLayer      INTEGER     Integer number corresponding to maximum layer index to extract activations. 
    #                           Usually last CONV layer before FC layers.

    layer_outputs = [layer.output for layer in model.layers[:maxLayer]]

    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(img_tensor)

    return activations


def visChannelsActs(model, img_tensor, maxLayer, images_per_row=16):

    # Get the names of the layers in the network
    layer_names = []
    for layer in model.layers:
        layer_names.append(layer.name)

    # Get the activations for the supplied image.
    activations = createActivationModel(model, img_tensor, maxLayer)

    # Now construct the grid of activations for plotting
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[1]

        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row
        display_grid = np.zeros(size * n_cols, images_per_row * size)

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]

                # Post process activation image for better plotting
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                # Save the channel activation image to the display grid
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
        
        # Plot the grid
        scale = 1. / size

        plt.figure(figsize=(scale * display_grid.shape[1],
                           scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')


def deprocess_image(x):
    ##
    #
    # Process tensor with floating points into an RGB image with values in the range of 0 to 255.

    x -= x.mean()
    x /= (x.std() + 1e-05)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x

def maximiseActivation(model, layer_name, filter_index, size=150, steps=40):
    ##
    # Function to maximise the activation of the nth filter in the ith layer of a network.
    #
    # model             OBJ     Compiled Keras model object of the network to be investigated.
    #
    # layer_name        STR     Name of the ith layer in the network to be activated.
    #
    # filter_index      INT     nth filter index of the ith layer to be activated.
    #

    # Build loss function that maximises activation of filter under consideration.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute gradient of input picture with regards to loss.
    grads = K.gradients(loss, model.input)[0]
    
    #with GradientTape() as t:
    #    t.watch(loss)
    #    grads = t.gradient(loss, model.input)

    # Normalisation trick, using L2 norm of the gradients
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-05)

    # Return the loss and gradients given the input picture
    iterate = K.function([model.input], [loss, grads])


    # Run the activation maximisation using random greyscale image as input
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    print('channel_ind:: ', filter_index,' ::', end='')

    step = 1.0
    for i in range(steps):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        print('>', end='')

    img = deprocess_image(input_img_data[0])
    print(':: Complete :: Final Loss ', loss_value,' ::')

    return img


def visMaxActivations(model, layer_name, size=64, margin=5, nx=8, figSize=(20,20)):
    # Plot a grid of activations for every channel in a layer within a network

    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

    print('[INFO] Starting computation of channel activations for layer:: ', layer_name)

    i_filter = 0
    for i in range(nx):
        for j in range(size // nx):

            print('[INFO] [Filter ',i_filter,' of ',nx * (size // nx),'] ',end='')
            filter_img = maximiseActivation(model=model, 
                                            layer_name=layer_name, 
                                            filter_index=i + (j * nx),
                                            size=size)
            
            horizontal_start = (i * size) + (i * margin)
            horizontal_end = horizontal_start + size
            vertical_start = (j * size) + (j * margin)
            vertical_end = vertical_start + size
            results[horizontal_start : horizontal_end,
                    vertical_start : vertical_end, :] = filter_img
            
            i_filter += 1
    
    # Plot the grid of activations for this layer
    plt.figure(figsize=figSize)
    plt.imshow(results)


def gradCAM(model, preproc_imag, last_conv_layer_name, model_train_img_index, plot_heatmap=False, figSize=(10,10)):

    output = model.output[:, model_train_img_index]

    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Gradient of the image class with regards to the output feature map of the last convolutional layer
    grads = K.gradients(output, last_conv_layer_name.output)[0]

    # Vectorize gradients, where every entry is the mean intensity of the gradient 
    # over a specific feature-map channel.
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([model.input], 
                         [pooled_grads, last_conv_layer.output[0]])

    # Values of pooled gradients and feature-maps given the sample image
    pooled_grads_value, conv_layer_output_value = iterate([preproc_imag])

    # Multiply each channel in feature-map by importance to image class
    for i in range(pooled_grads.shape[0]): # Check to make sure correct dims
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # Channel-wise mean of resulting feature-map is heatmap of class activation.
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    if plot_heatmap:
        heatmap  = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        plt.figure(figsize=figSize)
        plt.matshow(heatmap)

    return heatmap

def visGradCAM(img_path, heatmap, img_outname):

    # Import original image
    img = cv2.imread(img_path)

    # Resize heatmap to size of image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Apply heatmap to original image, and apply heatmap alpha factor, controls opacity
    superimposed_img = heatmap * 0.4 + img

    # Save the combined image to disk
    cv2.imwrite(img_outname, superimposed_img)