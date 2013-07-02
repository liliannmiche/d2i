"""Implementation of Optimal Pruned Extreme Learning Machine.
"""

from ibc_config import IBCConfig as cf
import numpy as np
from scipy.optimize import minimize
from mrsr import mrsr
import cPickle
import random


class ELM:

    def __init__(self):
        if cf._show_progress:        
            print "ELM started!"
        
    def __del__(self):
        if cf._show_progress:
            print "ELM finished!"

    def _project(self, X, neurons_dict):
        """Projects data to hidden layer H.

        X: training data (no labels)
        neurons_dict: keys are types of neurons, values are either an amount
                      of such neurons, or a matrix W with bias column included
        """        
        self.n_sampl = X.shape[0]
        H = np.empty((self.n_sampl,0), dtype=np.float64)        
        self.neurons_dict = {}
        #  W includes bias as an additional column
        #  Additional 1 input is added to data       
        X = np.hstack((X, np.ones((self.n_sampl, 1))))

        for key in neurons_dict.keys():
            if key in ['lin']:
                N = neurons_dict[key]
                if isinstance(N, (int, long, float, complex)):
                    W = np.eye(self.n_feat + 1)
                else:
                    W = neurons_dict[key]
                #  saving transformation matrix W
                self.neurons_dict['lin'] = W
                #  extending hidden layer matrix H     
                H0 = np.dot(X,W)
                H = np.hstack((H, H0))
                
            elif key in ['tanh']:
                N = neurons_dict[key]
                if isinstance(N, (int, long, float, complex)):
                    #  normal distributed coefficients with regularization 
                    #  to keep outputs in acceptable range
                    W = np.random.randn(self.n_feat+1, int(N)) / (int(N/10)**0.5)
                else:
                    W = neurons_dict[key]
                #  saving transformation matrix W
                self.neurons_dict['tanh'] = W
                #  extending hidden layer matrix H                
                H0 = np.tanh(np.dot(X,W))
                H = np.hstack((H, H0))                
                
            else:
                print "Unknows neuron type %s, skip" % key    
                
        #  return projection
        return H
     
     
    def e_LOO(self, H, Y, lmd=0):
        #  keeping number of samples low enough to fit into memory
        sampl_max = 1000
        if H.shape[0] > sampl_max:
            idx = range(H.shape[0])
            random.shuffle(idx)
            idx = idx[:sampl_max]
            H = H[idx,:]
            Y = Y[idx,:]
        if lmd == 0:
            pH = np.linalg.pinv(H)
            P = np.dot(H, pH)
        else:
            #  TROP-ELM stuff with lambda
            P = np.linalg.inv(np.dot(H.T, H) + np.eye(H.shape[1])*lmd)
            P = np.dot(H, P)
            P = np.dot(P, H.T)
        e1 = Y - np.dot(P, Y)
        e2 = np.ones((H.shape[0], )) - np.diag(P)
        e = e1/np.tile(e2, (Y.shape[1],1)).T
        #  mean error for multi-variate output, for each sample
        e = np.mean(np.abs(e), axis=1)
        return np.mean(e)        
     

    def train_basic(self, X, Y, neurons_dict):
        self.n_feat = X.shape[1]
        self.H = self._project(X, neurons_dict)    
        if cf._show_progress:
            print "basic: %d neurons, LOO error = %.4f" % (self.H.shape[1],
                                                           self.e_LOO(self.H, Y))        
        
        self.W1 = np.dot(np.linalg.pinv(self.H), Y)
        self.lmd = 0
        self.Yhat = np.dot(self.H, self.W1)

        
    def train_op(self, X, Y, neurons_dict, kmax=-1, step=-1):
        self.n_feat = X.shape[1]
        self.H = self._project(X, neurons_dict)    
        self.W1 = np.dot(np.linalg.pinv(self.H), Y)

        if kmax == -1:
            kmax = self.H.shape[1]
        if step == -1:
            step = max(kmax / 25, 1)
        E = np.ones((kmax,)) * float("+inf")
        print "basic: %d neurons, LOO error = %.4f" % (self.H.shape[1],
                                                       self.e_LOO(self.H, Y))        

        #  calculate LOO errors for different amount of centroids
        Rank = mrsr(Y, self.H, kmax=kmax)
        for i in range(0, kmax, step):  
            H = self.H[:, Rank[:i+1]]
            E[i] = self.e_LOO(H, Y)            

        #  find best sequence
        Rbest = Rank[:np.argmin(E)+1]

        #  offset is a starting number in current neuron dict item
        offset = 0        
        new_dict = {}
        for key in self.neurons_dict.keys():
            indmin = offset
            indmax = offset + self.neurons_dict[key].shape[1]
            offset = indmax
            rank_idx = Rbest[ (Rbest >= indmin) * (Rbest < indmax) ]
            rank_idx = rank_idx - indmin
            W0 = self.neurons_dict[key][:, rank_idx]            
            new_dict[key] = W0

        self.H = self._project(X, new_dict)
        self.W1 = np.dot(np.linalg.pinv(self.H), Y)
        self.lmd = 0
        print "pruned: %d neurons, LOO error = %.4f" % (self.H.shape[1],
                                                        self.e_LOO(self.H, Y))        
    

    def _train_trop(self, lmd):
        Y = self.temp_Y
        kmax = self.temp_kmax
        step = self.temp_step
        E = np.ones((kmax,)) * float("+inf")
        #  calculate LOO errors for different amount of centroids
        Rank = mrsr(Y, self.H, kmax=kmax)
        for i in range(0, kmax, step):        
            H = self.H[:, Rank[:i+1]]
            E[i] = self.e_LOO(H, Y, lmd)            

        self.Rbest = Rank[:np.argmin(E)+1]
        H = self.H[:, self.Rbest]
        return self.e_LOO(H, Y, lmd)


    def train_trop(self, X, Y, neurons_dict, kmax=-1, step=-1):
        self.n_feat = X.shape[1]
        self.H = self._project(X, neurons_dict)    
        self.W1 = np.dot(np.linalg.pinv(self.H), Y)
        self.lmd = 0
        if kmax == -1:
            kmax = self.H.shape[1]
        if step == -1:
            step = max(kmax / 25, 1)
        self.temp_Y = Y
        self.temp_kmax = kmax
        self.temp_step = step
        print "basic: %d neurons, LOO error = %.4f" % (self.H.shape[1],
                                                        self.e_LOO(self.H, Y))        

        res = minimize(self._train_trop, 0.5, method="Nelder-Mead")
        if not res.success:
            print "Lambda optimization failed, using basic results"
            print "(or try to re-run the function)" 
            return False
        lmd = res.x
        Rbest = self.Rbest

        #  re-compiling W0 matrix according to ranking outputs
        #  offset is a starting number in current neuron dict item
        offset = 0        
        new_dict = {}
        for key in self.neurons_dict.keys():
            indmin = offset
            indmax = offset + self.neurons_dict[key].shape[1]
            offset = indmax
            rank_idx = Rbest[ (Rbest >= indmin) * (Rbest < indmax) ]
            #  if we have some neurons of that type
            if len(rank_idx) > 0:
                print "%s neurons: %d" % (key, len(rank_idx))
                rank_idx = rank_idx - indmin
                W0 = self.neurons_dict[key][:, rank_idx]            
                new_dict[key] = W0

        self.H = self._project(X, new_dict)
        self.W1 = np.dot(np.linalg.pinv(self.H), Y)
        self.lmd = lmd
        print "final: %d neurons, LOO error = %.4f, lambda = %f"\
            % (self.H.shape[1], self.e_LOO(self.H, Y, lmd), lmd)      
        return True
        
    def run(self, X, Y=[]):
        H = self._project(X, self.neurons_dict)
        Yhat = np.dot(H, self.W1)
        if Y != []:
            e = self.e_LOO(H, Y, self.lmd)
            return (Yhat, e)
        return Yhat

    def get_param(self):
        param = {}
        param['W0'] = self.neurons_dict
        param['W1'] = self.W1
        param['lmd'] = self.lmd
        return param
        
    def set_param(self, param):
        self.neurons_dict = param['W0']
        self.W1 = param['W1']
        self.lmd = param['lmd']
        

if __name__=="__main__":
    (X,Y) = cPickle.load(open("iris.pkl","rb"))
    
    #D = cPickle.load(open("/home/akusoka1/Desktop/data2elm.pkl","rb"))
    #X = D['Xr']
    #Y = D['Yr']
    elm = ELM()
    elm.train_trop(X, Y, {'lin':1, 'tanh':50})
    #(Yhat,e) = elm.run(D['Xs'], D['Ys'])
    #print float(np.sum(np.argmax(Yhat,1) == np.argmax(D['Ys'],1))) / D['Ys'].shape[0]










    