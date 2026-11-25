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
        P_use = self.P.copy()
        sigma2_use = self.params["sigma2"]
        if e> 0:
            P_use[1,1] = self.P[1,1] * self.params["rho"]
            sigma2_use = sigma2_use * self.params["lam"]
        PHt= P_use @ self.H
        D= self.H @ PHt + sigma2_use
        K= PHt / D
        self.mu= self.mu + K * e
        self.P= P_use - np.outer(K, PHt)
    
    def new_task(self):
        self.mu[1] = 0.0
        self.P[0,1] = 0.0
        self.P[1,0] = 0.0
        self.P[1,1] = self.params['s2_d']
