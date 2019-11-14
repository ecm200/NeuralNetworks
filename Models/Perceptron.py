import numpy as np

class Perceptron:
    
    def __init__(self, N, alpha=0.1):
        # Init weight matrix and store learning rate
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha
        
        
    def step(self, x):
        
        return 1 if x > 0 else 0
    
    def fit(self, X, y, epochs=10):
        # Insert column for bias
        X = np.c_[X,np.ones(X.shape[0])]
        
        print('[INFO] Begin training loop...', end='')
        # Loop over epoch
        for epoch in np.arange(0, epochs):
            
            print(str(epoch)+'..', end='')
            
            # Loop over each data point
            for (x, target) in zip(X, y):
                
                # Take dot product between input features and weight matrix.
                # Pass output through step function
                p = self.step(np.dot(x, self.W))
                
                # Only perform update if prediction is wrong
                if p != target:
                    error = p - target
                    self.W += -self.alpha * error * x
        print('Training complete')
                    
    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)
        
        if addBias:
            # Insert column of 1's as Bias column
            X = np.c_[X, np.ones(X.shape[0])]
            
        # Take dot product between input features and weight matrix, then pass through step funciton
        return self.step(np.dot(X, self.W))
    
    def test(self, X, y):
        
        print('[INFO] Testing Perceptron')

        for (x, target) in zip(X, y):
            pred = self.predict(x)
            print('[INFO] data={}, ground-truth={}, pred={}'.format(x, target[0], pred))