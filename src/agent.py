import numpy as np


class Agent:
    def __init__(self,params):
        self.params = params
        self.mu = np.array([params["m_g0"],0.0])
        self.P= np.diag([params['s2_g'],params['s2_d']])
        self.H= np.array([1.0,1.0]) 
    
    def predict(self):
        return self.H @ self.mu
    
    def update(self,y):
        e= y - self.predict()
        PHt= self.P @ self.H
        D= self.H @ PHt + self.params['sigma2']
        K= PHt / D
        self.mu= self.mu + K * e
        self.P= self.P - np.outer(K, PHt)
    
    def new_task(self):
        self.mu[1] = 0.0
        self.P[0,1] = 0.0
        self.P[1,0] = 0.0
        self.P[1,1] = self.params['s2_d']
