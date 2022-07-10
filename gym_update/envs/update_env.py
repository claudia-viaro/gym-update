#packages environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas.testing as tm
import math
from sklearn.linear_model import LogisticRegression
from scipy.stats import truncnorm

#Gym environment - continuous

class UpdateEnv(gym.Env):
  def __init__(self):
    self.size = 2000     

    #set range for action space
    self.high_th = np.array([2, 2, 2])
   
    #set ACTION SPACE
    self.action_space = spaces.Box(
            low = np.float32(-self.high_th),
            high = np.float32(self.high_th))
    
    #set range for obs space    
    self.min_Xas=np.array([0, 0])
    self.max_Xas=np.array([math.inf, math.inf])
    
    #set OBSERVATION SPACE
    #it is made of values for Xa, Xs for each observation
    self.observation_space = spaces.Box(low=np.float32(self.min_Xas),
                                        high=np.float32(self.max_Xas))
        
    #set an initial state
    self.state=None 
    
    self.init_actions = [0.1, 0.1, 0.1]

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]    
 
  def intervention(self, Xa, rho):
    # Xa is Xa_e(0))
    # rho is rho_e-1(Xs_e(0), Xa_e(0))
    g = ((Xa) + 0.5*((Xa)+np.sqrt(1+(Xa)**2)))*(1-rho**2) + ((Xa) - 0.5*((Xa)+np.sqrt(1+(Xa)**2)))*(rho**2)
    return g
    
#take an action with the environment
  def step(self, action):
    
    done = False
    
    #-----------------------------------------------------------------------------------
    # e=e, t=0
    # observe new patients (Xs_e(0), Xa_e(0))
    pat_e0= np.hstack([np.ones((self.size, 1)),truncnorm.rvs(a=0, b= math.inf,size=(self.size,2))]) #shape (size, 3), (1, Xs, Xa)
    
    # compute rho_0(Xs_e(0), Xa_e(0)) - we're using covariates at e and thetas at e-1
    rho_0 = (1/(1+np.exp(-(np.matmul(pat_e0, self.init_actions[:, None])))))[:, 0]  #prob of Y=1. # (sizex3) x (3x1) = (size, 1)
    
    # decide an intervention, use rho_0, Xa_1(0)
    g_e = self.intervention(pat_e0[:, 2], rho_0)
    
    #-----------------------------------------------------------------------------------
    # e=1, t=1
    #update Xa_1(0) to Xa_1(1) with intervention
    Xa = ge # size
    # predict f_1 = E[Y_1|X_1(1)] 
    f_e = 1/(1+ np.exp(-pat_e0[:, 1]-Xa))
    
    # observe Y_1(1)
    Y_1 = np.random.binomial(1, 0.2, (self.size, 1))
    pat_e1 = np.hstack([Y_1, np.reshape(pat_e0[:, 1], (size, 1)), np.reshape(Xa, (size, 1))]) #shape (size, 3), (Y, Xs, Xa)                
    
    # use actions (thetas) from environment                  
    rho_e = (1/(1+np.exp(-(np.matmul(pat_00, action[:, None])))))  #prob of Y=1 # (sizex3) x (3x1) = (size, 1 
    
    # link drop in replacement of rho
    
                          
    #-----------------------------------------------------------------------------------
    # naive result, uses variables without intervention. use pat_e0, Y_e
    pat_naive =  np.hstack([Y_1, np.reshape(pat_e0, (size, 2))])  
    model_naive = LogisticRegression().fit(pat_naive[:, 1:3], np.ravel(pat_naive[:, 0].astype(int)))                        
    thetas_naive = np.array([model_naive.intercept_[0], model_naive.coef_[0,0] , model_naive.coef_[0,1]]) #thetas_n[0]: intercept; thetas_n[1]: coef for Xs, thetas_n[2] coef for Xa
    pat_naive00 = np.hstack([np.ones((self.size, 1)), np.reshape(pat_e0, (size, 2))])
    rho_naive = (1/(1+np.exp(-(np.matmul(pat_naive00, thetas_naive[:, None])))))  #prob of Y=1 # (sizex3) x (3x1) = (size, 1)  
                            
    #-----------------------------------------------------------------------------------                        
    # we return:
    # - f_1 = E[Y_1|X_1(1)]                      
    # - rho_naive, that is E[Y_1|X_1(0)]                                               
    
    self.patients = pat_e1[:, 1:3]
    
    
      
    #check if horizon is over, otherwise keep on going
    if np.mean(rho_e) >= 0.3:
      done = True
    else:
      done = False
        
    
    
    
  
    return {"patients": self.patients, "f_e": f_e, "naive_patients": pat_naive[:, 1:3], "naive_rho": rho_naive,"done": done}
    
    
#reset state and horizon    
  def reset(self):  
                          
    # e=0, t=0
    # observe self.patients (from reset function, self.patients) (Xs(0), Xa(0))
    self.patients = truncnorm.rvs(a=0, b= math.inf,size=(self.size,2)) #shape (size, 2), 1st columns is Xs, second is Xa                        
    
    #-----------------------------------------------------------------------------------
    #e=0, t=1    
    # observe same patients (Xs(1), Xa(1))=(Xs(0), Xa(0))
    # predict f_0 = E[Y_0|X_0]                      
    f_0 = 1/(1+ np.exp(-self.patients[:, 0]-self.patients[:, 1]))                      

    # observe Y(1)    
    Y_0 = np.random.binomial(1, 0.2, (self.size, 1))
    pat_01 = np.hstack([Y_0, self.patients]) # Y, Xs, Xa
    
    # decide on rho_0
    # model_0 = LogisticRegression().fit(pat_01[:, 1:3], np.ravel(pat_01[:, 0].astype(int)))
    # thetas_0 = np.array([model_0.intercept_[0], model_0.coef_[0,0] , model_0.coef_[0,1]]) #thetas0[0]: intercept; thetas0[1]: coef for Xs, thetas0[2] coef for Xa
    # pat_00 = np.hstack([np.ones((self.size, 1)), self.patients]) #1, Xs, Xa
    # rho_0 = (1/(1+np.exp(-(np.matmul(pat_00, thetas_0[:, None])))))  #prob of Y=1 # (sizex3) x (3x1) = (size, 1)
    
    
    # decide on initial actions, or simply use some values
    # initial_actions = thetas_0
       
    
    # i don't really think there's need for initial actions any longer
    # f_0 and rho_0 are the same at e=0                                                        
    return {"f_0": f_0, "patients": self.patients}
