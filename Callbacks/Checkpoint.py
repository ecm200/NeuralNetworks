from tensorflow.keras.callbacks import ModelCheckpoint

def saveBestModel(saveModelPath='./model-checkpoint.h5', monitor='val_accuracy', saveWeightsOnly=False, 
                    mode='max', save_freq='epoch', verbose=1):

    return ModelCheckpoint(filepath=saveModelPath, monitor=monitor, verbose=verbose, 
                                 save_best_only=True, save_weights_only=saveWeightsOnly, mode=mode, save_freq=save_freq)