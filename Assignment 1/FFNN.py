import numpy as np
# import pandas as pd
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
np.random.seed(0)


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
            exps = np.exp(x[i]- np.max(x[i]))
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
        return np.mean(loss)
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
        return np.mean(loss)    
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

    def fit(self,X,y, Xval, yval):
      for ep in range(self.epochs):
        X, y = shuffle(X, y, random_state=ep)
        for i in range(0,X.shape[0],self.batch_size):
          x_batch = X[i:i + self.batch_size]
          y_batch = y[i:i + self.batch_size]
          self.fit_batch(x_batch, y_batch,(i/self.batch_size))
        y_true = self.process_y(y)
        acc_t=(np.mean(self.predict(X).argmax(axis=-1)==y_true.argmax(axis=-1)))
        y_pval = self.predict(Xval)
        y_tval = self.process_y(yval)
        acc_v=(np.mean(y_pval.argmax(axis=-1)==y_tval.argmax(axis=-1)))
        #print(y_true)
        print("Epoch: "+str(ep+1)+", Train accuracy: "+str(acc_t)+" Val accuracy : "+str(acc_v))
    
    def fit_epoch(self, X, y, epoch):
        X, y = shuffle(X, y, random_state=epoch)
        for i in range(0,X.shape[0],self.batch_size):
          x_batch = X[i:i + self.batch_size]
          y_batch = y[i:i + self.batch_size]
          self.fit_batch(x_batch, y_batch,(i/self.batch_size))
        y_true = self.process_y(y)
        acc_t=(np.mean(self.predict(X).argmax(axis=-1)==y_true.argmax(axis=-1)))
        #print(y_true)
        print("Epoch: "+str(epoch+1)+", Train accuracy: "+str(acc_t))

    def predict(self,X):
      y_pred = self.forward(X)
      return y_pred
    
    def evaluate(self, y_pred, y_true):
        y_true = self.process_y(y_true)
        acc=(np.mean(y_pred.argmax(axis=-1)==y_true.argmax(axis=-1)))
        loss = self.loss_function.loss(y_pred,y_true)
        l2reg = 0
        for layer in self.network:
            l2reg += self.wd*np.sum(np.square(layer.w))
        l2reg = l2reg/(2.0 * y_true.shape[0])
        return acc, loss+l2reg
    
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

class RMSOptim():
    def __init__(self, eta=0.1, beta1=0.9, eps=1e-8, wd=0):
        self.v_w, self.v_b = 0, 0
        self.beta1 = beta1
        self.eps = eps
        self.eta = eta
        self.wd = wd
        self.name = 'rms'

    def update(self, w, b, dw, db, t=0):
        ## dw, db are from current minibatch
        ## momentum beta 1
        self.v_w = self.beta1*self.v_w + (1-self.beta1)*(dw**2)
        self.v_b = self.beta1*self.v_b + (1-self.beta1)*(db**2)

        ## update weights and biases
        w = w - (self.eta/(np.sqrt(self.v_w+self.eps)))*dw - self.eta*(self.wd)*(w)
        b = b - (self.eta/(np.sqrt(self.v_b+self.eps)))*db
        return w, b

class adamOptim():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, wd=0):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.wd = wd
        self.name = 'adam'
    def update(self, t, w, b, dw, db):
        ## dw, db are from current minibatch
        ## momentum beta 1
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        self.m_db = self.beta1*self.m_db + (1-self.beta1)*db

        ## rms beta 2
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db**2)

        ## bias correction
        m_dw_hat = self.m_dw/(1-self.beta1**(t+1))
        m_db_hat = self.m_db/(1-self.beta1**(t+1))
        v_dw_hat = self.v_dw/(1-self.beta2**(t+1))
        v_db_hat = self.v_db/(1-self.beta2**(t+1))

        ## update weights and biases
        w = w - (self.eta/(np.sqrt(v_dw_hat+self.epsilon)))*(m_dw_hat) - self.eta*(self.wd)*(w)
        b = b - (self.eta/(np.sqrt(v_db_hat+self.epsilon)))*(m_db_hat)
        return w, b

class nadamOptim():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, wd = 0):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.wd = wd
        self.name='nadam'

    def update(self, t, w, b, dw, db):
        ## dw, db are from current minibatch
        ## momentum beta 1
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        self.m_db = self.beta1*self.m_db + (1-self.beta1)*db

        ## rms beta 2
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db**2)

        ## bias correction
        m_dw_hat = self.m_dw/(1-self.beta1**(t+1))
        m_db_hat = self.m_db/(1-self.beta1**(t+1))
        v_dw_hat = self.v_dw/(1-self.beta2**(t+1))
        v_db_hat = self.v_db/(1-self.beta2**(t+1))

        ## nesterov
        m_dw_m = self.beta1*m_dw_hat + ((1-self.beta1)*(dw))/(1-self.beta1**(t+1))
        m_db_m = self.beta1*m_db_hat + ((1-self.beta1)*(db))/(1-self.beta1**(t+1))

        ## update weights and biases
        w = w - (self.eta/(np.sqrt(v_dw_hat+self.epsilon)))*(m_dw_m) - self.eta*(self.wd)*(w)
        b = b - (self.eta/(np.sqrt(v_db_hat+self.epsilon)))*(m_db_m)
        return w, b




# # Get training and testing vectors 
# (trainX, trainy), (testX, testy) = fashion_mnist.load_data()

# trainX = trainX.reshape(60000, 784)/255.0
# testX = testX.reshape(10000, 784)/255.0

# X_train, X_val, y_train, y_val = train_test_split(trainX, trainy, test_size=0.1, random_state=0)

# Training and testing
def traintest():
    neuronlist = []
    for i in range(3):
        neuronlist.append([64, 'tanh'])
    parameters = dict(input_size = 784, output_size = 10, neuronlist = neuronlist,
                  batch_size = 32, epochs = 5, optimizer = 'rms',
                  learning_rate = 0.001, wb_initializer = 'xavier', weight_decay = 0.0005,
                  loss_function = 'crossentropy')
        
    fnn = nn(**parameters)
    (trainX, trainy), (testX, testy) = fashion_mnist.load_data()
    trainX = trainX.reshape(60000, 784)/255.0
    testX = testX.reshape(10000, 784)/255.0
    X_train, X_val, y_train, y_val = train_test_split(trainX, trainy, test_size=0.1, random_state=0)

    for epoch in range(5):
        fnn.fit_epoch(X_train, y_train, epoch)
        y_vpred = fnn.predict(X_val)
        val_acc, val_loss = fnn.evaluate(y_vpred, y_val)
        print("Validation accuracy: "+str(val_acc)+', Val loss:' +str(val_loss))
        y_tpred = fnn.predict(testX)
        acc, loss = fnn.evaluate(y_tpred, testy)
        print("Test accuracy: "+str(acc)+', test loss:' +str(loss))

traintest()


# #############################################
# # HYPERPARAMETER TUNINING USING WANDB SWEEP #
# #############################################

# import wandb
# wandb.login()

# sweep_config = {
#     'method': 'RANDOM', #grid, random, bayes
#     'metric': {
#       'name': 'val_loss',
#       'goal': 'minimize'   
#     },
#     'parameters': {
#         'epochs': {
#             'values': [5, 10]
#         },
#         'num_layers': {
#             'values': [3, 4, 5]
#         }
#         'layer_size': {
#             'values': [16, 32, 64]
#         },
#         'weight_decay': {
#             'values': [0, 0.0005, 0.5]
#         },
#         'learning_rate': {
#             'values': [1e-3, 1e-4]
#         },
#         'optimizer': {
#             'values': ['nadam', 'adam', 'rms', 'nag', 'mgd', 'sgd']
#         },
#         'batch_size': {
#             'values': [64, 32, 16]
#         },
#         'wb_initializer':{
#             'values': ['random', 'xavier']
#         },
#         'activation': {
#             'values': ['sigmoid', 'relu', 'tanh']
#         }
#     }
# }

# sweep_id = wandb.sweep(sweep_config, project="cs6910-a1")

# def train(config=None):
#     with wandb.init(config = config):
#         config = wandb.config
#         neuronlist = []
#         for i in range(config.num_layers):
#             neuronlist.append([config.layer_size, config.activation])
#         parameters = dict(input_size = 784, output_size = 10, neuronlist = neuronlist,
#                   batch_size = config.batch_size, epochs = config.epochs, optimizer = config.optimizer,
#                   learning_rate = config.learning_rate, wb_initializer = config.wb_initializer, weight_decay = config.weight_decay,
#                   loss_function = 'crossentropy')
#         wandb.run.name = 'hn-'+str(config.num_layers)+'_hs-'+str(config.layer_size)+'_a-'+config.activation+'_bs-'+str(config.batch_size)+'_o-'+config.optimizer
#         fnn = nn(**parameters)
#         (trainX, trainy), (testX, testy) = fashion_mnist.load_data()
#         trainX = trainX.reshape(60000, 784)/255.0
#         testX = testX.reshape(10000, 784)/255.0
#         X_train, X_val, y_train, y_val = train_test_split(trainX, trainy, test_size=0.1, random_state=0)
#         for epoch in range(config.epochs):
#             fnn.fit_epoch(X_train, y_train, epoch)
#             y_vpred = fnn.predict(X_val)
#             val_acc, val_loss = fnn.evaluate(y_vpred, y_val)
#             y_tpred = fnn.predict(testX)
#             acc, loss = fnn.evaluate(y_tpred, testy)
#             wandb.log({'val_loss': val_loss, 'val_accuracy': val_acc, 
#                        'loss': loss, 'accuracy': acc, 'epoch': epoch+1})

# # Running wandb sweep for 25 runs
# wandb.agent(sweep_id, train, count=25)