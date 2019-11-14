import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

import numpy as np

def plotLearnRates(modelHistory, epochs=100, figSize=(12,8), saveFig=False, 
                   path='learnungCurve.png'):
    
    plt.style.use("ggplot")
    plt.figure(figsize=figSize)
    plt.plot(np.arange(0, epochs), modelHistory.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), modelHistory.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), modelHistory.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), modelHistory.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    if saveFig:
        plt.savefig(path)
    plt.show()
