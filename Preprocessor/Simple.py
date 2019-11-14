import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # Store target image width, height, interpolation when used in resizing.
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self, image):
        # Resize image to fixed pixel size, ignore aspect ratio.
        return cv2.resize(image, (self.width, self.height),
                         interpolation=self.inter)