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
    def __init__(self,inp,out,act,wbinit, optimizer, eta, wd):
      self.prev_n=inp
      self.curr_n=out
      self.activation=act
      self.wb_initializer=wbinit
      self.wd = wd
      self.grad_w, self.grad_b = 0,0
      if optimizer=='sgd':
        self.optimizer = SGDOptim(eta = eta, wd = self.wd)
      elif optimizer=='mgd':
        self.optimizer = MomentGDOptim(eta = eta, wd = self.wd)
      elif optimizer=='nag':
        self.optimizer = NAGDOptim(eta = eta, wd = self.wd)
      elif optimizer=='rms':
        self.optimizer = RMSOptim(eta = eta, wd = self.wd)
      elif optimizer=='adam':
        self.optimizer = adamOptim(eta = eta, wd = self.wd)
      elif optimizer=='nadam':
        self.optimizer = nadamOptim(eta = eta, wd = self.wd)
      self.initialize_wb()
    
    def initialize_wb(self):
      self.w = self.wb_initializer.init_w(self.prev_n,self.curr_n)
      self.b = np.zeros(self.curr_n)

    def get_grad_w(self,a,b):
      #c=np.einsum("ij,ik->ikj",a,b)
      #d=np.mean(c,axis=0)
      c = np.dot(b.T, a)
      return c

    def get_grad_b(self,a):
      c=np.mean(a,axis=0)
      return c

    def get_grad_h(self,a,b):
      c=np.einsum("ij,kj->ki",a,b)
      return c

    def get_grad_a(self,a,b):
      c=a*b
      return c
    
    def get_derivs(self,a,act):
      return act.grad(a)
    
    def frwd(self,inputs):
      self.input=inputs
      self.a = np.dot(self.input,self.w)+self.b
      self.h = self.activation.activate(self.a)
      return self.h

    def bkwd(self,grad_a,prev_layer_a,prev_act,i):
      self.grad_w += self.get_grad_w(grad_a,self.input)
      #self.grad_b=np.mean(grad_a,axis=0)
      self.grad_b += np.mean(grad_a, axis=0) * self.input.shape[0] 
      if i==0:
        return grad_a
      grad_a=np.dot(grad_a,self.w.T)*prev_act.grad(prev_layer_a)
      return grad_a
    
    def update_wb(self,t):
      self.w, self.b = self.optimizer.update(w = self.w, b=self.b, dw=self.grad_w, db=self.grad_b, t=t)
      self.grad_w , self.grad_b = 0,0

    def partial_wb(self):
      v_w, v_b = self.optimizer.partial()
      self.w = self.w - v_w
      self.b = self.b - v_b
    def get_act(self):
      return self.activation
    def get_a(self):
      return self.a


class nn():
    def __init__(self,input_size,output_size,neuronlist,batch_size,epochs,optimizer,loss_function,learning_rate,wb_initializer, weight_decay):
      self.network=[]
      self.batch_size=batch_size
      self.learning_rate=learning_rate
      self.epochs=epochs
      self.wd = weight_decay

      fl=[[input_size,'dummystr']]+neuronlist+[[output_size,'softmax']]

      self.optimizer = optimizer

      if loss_function=='crossentropy':
        self.loss_function=crossEntropy()
      elif loss_function=='meansq':
        self.loss_function=meanSq()
      if wb_initializer=='xavier':
        self.wb_initializer=xavier()
      elif wb_initializer=='random':
        self.wb_initializer=randwb()

      for i in range(len(fl)-1):
        actstr=fl[i+1][1]
        if actstr=='sigmoid':
          activation=sigmoid()
        elif actstr=='tanh':
          activation=tanh()
        elif actstr=='relu':
          activation=relu()
        elif actstr=='softmax':
          activation=softmax()
        self.network.append(layer(inp=fl[i][0],out=fl[i+1][0],act=activation,wbinit=self.wb_initializer, optimizer = self.optimizer, eta=self.learning_rate, wd=self.wd))

    def forward(self,X):
      for layer in self.network:
        X=layer.frwd(X)
      return X
      
    def process_y(self,y):
      y_t = np.zeros((y.shape[0], 10))
      for i in range(y.shape[0]):
        y_t[i][y[i]]=1
      return y_t

    def fit_batch(self,X,y,t):
      y_pred=self.forward(X)
      y_true=self.process_y(y)
      loss=self.loss_function.loss(y_pred,y_true)
      grad_a=self.loss_function.loss_grad(y_pred,y_true)
      for i in range(len(self.network)-1,-1,-1):
        layer=self.network[i]
        if layer.optimizer.name=='nag':
          layer.partial_wb()
        else:
          pass
        if i!=0:
          grad_a=layer.bkwd(grad_a,self.network[i-1].get_a(),self.network[i-1].get_act(),i)
        else:
          grad_a=layer.bkwd(grad_a,self.network[0].get_a(),self.network[0].get_act(),i)
        layer.update_wb(t)

    def fit(self,X,y):
      for ep in range(self.epochs):
        for i in range(0,X.shape[0],self.batch_size):
          x_batch = X[i:i + self.batch_size]
          y_batch = y[i:i + self.batch_size]
          self.fit_batch(x_batch, y_batch,(i/self.batch_size))
        y_true = self.process_y(y)
        acc_log=(np.mean(self.predict(X).argmax(axis=-1)==y_true.argmax(axis=-1)))
        #print(y_true)
        print(f"Epoch: {ep+1}, Accuracy: {acc_log}")
    
    def predict(self,X):
      y_pred = self.forward(X)
      #print(y_pred)
      #prob=self.network[-1].h
      return y_pred
    
    def evaluate(self, y_pred, y_true):
        y_true = self.process_y(y_true)
        acc=(np.mean(y_pred.argmax(axis=-1)==y_true.argmax(axis=-1)))
        return acc
    
## WEIGHT INITIALIZER
class randwb():
  def __init__(self):
    pass
  def init_w(self, prev_n, curr_n):
    return np.random.randn(prev_n, curr_n)

class xavier():
  def __init__(self):
    pass
  def init_w(self, prev_n, curr_n):
    return np.random.normal(0,np.sqrt(6/(prev_n+curr_n)),(prev_n, curr_n))

##  OPTIMIZERS
class SGDOptim():
    def __init__(self, eta=0.01, wd = 0):
        self.eta = eta
        self.wd = wd
        self.name = 'sgd'

    def update(self, w, b, dw, db, t=0):
        ## dw, db are from current minibatch
        ## update weights and biases
        w = w - self.eta*(dw) - self.eta*(self.wd)*(w)
        b = b - self.eta*(db) 
        return w, b

class MomentGDOptim():
    def __init__(self, eta=0.01, gamma=0.9, wd=0):
        self.v_w, self.v_b = 0, 0
        self.gamma = gamma
        self.eta = eta
        self.wd = wd
        self.name = 'mgd'

    def update(self, w, b, dw, db, t=0):
        ## dw, db are from current minibatch
        ## momentum 
        self.v_w = self.gamma*self.v_w + self.eta*dw
        self.v_b = self.gamma*self.v_b + self.eta*db

        ## update weights and biases
        w = w - self.v_w - self.eta*(self.wd)*(w)
        b = b - self.v_b 
        return w, b

class NAGDOptim():
    def __init__(self, eta=0.01, gamma=0.9, wd=0):
        self.v_w, self.v_b = 0, 0
        self.prev_vw, self.prev_vb = 0,0
        self.gamma = gamma
        self.eta = eta
        self.wd = wd
        self.name = 'nag'

    def partial(self):
        self.v_w = self.gamma*self.prev_vw
        self.v_b = self.gamma*self.prev_vb
        return self.v_w, self.v_b

    def update(self, w, b, dw, db, t=0):
        ## dw, db are from current minibatch
        ## momentum 
        self.v_w = self.gamma*self.prev_vw + self.eta*dw
        self.v_b = self.gamma*self.prev_vb + self.eta*db
        
        ## update weights and biases
        w = w - self.eta*dw - self.eta*(self.wd)*(w)
        b = b - self.eta*db

        ##
        self.prev_vw, self.prev_vb = self.v_w, self.v_b
        return w, b