import cv2
import numpy as np
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None, labelsFromPath=True):
        # Store image preprocessors
        self.preprocessors = preprocessors

        # Store if getting labels from path of images
        self.labelsFromPath = labelsFromPath
        
        # If the preprocessors are None, init as empty list
        if self.preprocessors is None:
            self.preprocessors = []

            
    def load(self, imagePaths, verbose=-1):
        # Init list of features and labels
        data = []
        if self.labelsFromPath:
            labels = []
        
        # Loop over input images
        for (i, imagePath) in enumerate(imagePaths):
            # Load image and extract class label assuming our path has specified format.
            image = cv2.imread(imagePath)
            if self.labelsFromPath:
                label = imagePath.split(os.path.sep)[-2]
            
            if self.preprocessors is not None:
                # Loop over preprocessors
                for p in self.preprocessors:
                    image = p.preprocess(image)
                    
            # Treat processed image as feature vector by updating data list by labels
            data.append(image)
            if self.labelsFromPath:
                labels.append(label)
            
            # Show an update every 'verbose' images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print('[INFO] processed {}/{}'.format(i + 1, len(imagePaths)))
                
        return (np.array(data), np.array(labels)) if self.labelsFromPath else np.array(data)