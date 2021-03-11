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


## LOSS FUNCTIONS
class crossEntropy():
    def __init__(self):
        pass
    def loss(self,y_pred,y_true):
        n = len(y_pred)
        y_true_1 = [np.where(temp == 1) for temp in y_true]
        loss = np.array([-np.log(y_pred[i][y_true_1[i]]) for i in range(n)])
        return loss
    def loss_grad(self,y_pred,y_true):
        loss_grad = y_pred - y_true
        return loss_grad
     
class meanSq():
    def __init__(self):
        pass
    def solve(self,a,b):
        c = a - b
        d = c*a
        s = np.sum(d)
        return d - s*a    
    def loss(self,y_pred,y_true):
        loss = 0.5*np.sum(np.square(y_pred-y_true),axis=1)
        return loss    
    def loss_grad(self,y_pred,y_true):
        n = len(y_pred)
        loss_grad = np.array([self.solve(y_pred[i],y_true[i]) for i in range(n)])
        return loss_grad

class layer():
class nn():
