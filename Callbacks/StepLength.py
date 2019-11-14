import numpy as np

def step_decay(epoch):

    # Init base initial learning rate, drop factor and epochs to drop
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5

    # Compute learning rate for current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    return float(alpha)