import imutils
import cv2


class AspectAwarePreprocessor:

    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter


    def preprocess(self, image):
        # Grab image dimensions of image and then initialise the deltas to use when cropping
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        # If width is smaller than height, then resize along width, then update deltas 
        # to crop height to desired dimension
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)

        # Otherwise, the height is smaller than the width so resize along the height 
        # and then update deltas to crop along the width
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        # Now images have been resized, we need to re-grab width and height, 
        # followed by performing the crop.
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]

        # Resize the image to the provided spatial dimensions to ensure our output 
        # image is always a fixed size.
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
        