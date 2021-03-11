import numpy as np
import pandas as pd


# ACTIVATION FUNCTIONS
class sigmoid():
    def __init__(self):
        pass
    def activate(self, x):
        return 1/(1 + np.exp(-x))

    def grad(self, x):
        return self.activate(x)*(1 - self.activate(x))

class relu():
    def __init__(self):
        pass
    def activate(self, x):
        return np.maximum(x,0)
  
    def grad(self, x):
        x[x>0] = 1
        x[x<0] = 0
        return x

class tanh():
    def __init__(self):
        pass
    def activate(self, x):
        return np.tanh(x)

    def grad(self, x):
        return (1 -(np.tanh(x)**2))

class softmax():
    def __init__(self):
        self.name='softmax'
        pass
    def activate(self, x):
        out = np.zeros(x.shape)
        for i in range(0, x.shape[0]):
            exps = np.exp(x[i])
            out[i] = exps / np.sum(exps)        
        return out

    def grad(self, x):  
        pass    

class layer():
class nn():
