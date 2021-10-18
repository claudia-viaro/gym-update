# -*- coding: utf-8 -*-
"""update_env.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LPsUNmSmWMzg0OQ7VbuMpuW02FTtrGWP
"""

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#packages environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
import pandas.testing as tm
import math
from sklearn.linear_model import LogisticRegression
from scipy.stats import truncnorm

#Gym environment - continuous

class UpdateEnv(gym.Env):
  def __init__(self):
    self.size = 2000     

    #set range for action space
    self.high_th = np.array([4, 4, 4])
   
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
                                        high=np.float32(self.max_Xas),
                                        dtype=np.float32)
        
    #set an initial state
    self.state=None 

    #introduce some length
    self.horizon=2000 

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]    

#take an action with the environment
  def step(self, action):
    #e=0, t=0
    
    #e=0, t=1
    
    #see actions
    theta0, theta1, theta2 = action    
        
    #see patients
    patients1= np.hstack([np.ones((self.size, 1)), self.patients]) #shape (50, 3), 1st column of 1's, 2nd columns Xs, 3rd column Xa
    rho1 = (1/(1+np.exp(-(np.matmul(patients1, action[:, None])))))  #prob of Y=1  # (sizex3) x (3x1) = (size, 1)
    rho1 = rho1.squeeze() # shape: size, individual risk
    
    #e=1, t=0
    #see new patients
    patients2 = truncnorm.rvs(a=0, b= math.inf,size=(self.size,2)) #shape (size, 2), 1st columns is Xs, second is Xa
    Xa = patients2[:, 2] # shape: size
    g2 = ((Xa) + 0.5*((Xa)+np.sqrt(1+(Xa)**2)))*(1-rho1**2) + ((Xa) - 0.5*((Xa)+np.sqrt(1+(Xa)**2)))*(rho1**2)
    
    #e=1, t=1
    Xa = g2 # size
    Y = np.random.binomial(1, 0.2, (self.size, 1))
    patients3 = np.hstack([Y, np.reshape(patients2[:, 0], (self.size,1)), np.reshape(Xa, (self.size,1))]) #Y, Xs, Xa
    model2 = LogisticRegression().fit(patients3[:, 1:3], np.ravel(patients3[:, 0].astype(int)))
    thetas2 = np.array([model2.intercept_[0], model2.coef_[0,0] , model2.coef_[0,1]]) #thetas2[0]: intercept; thetas2[1]: coef for Xs, thetas2[2] coef for Xa
    patients4 = np.hstack([np.ones((self.size, 1)), patients3[:, 1:3]]) #1, Xs, Xa
    rho3 = (1/(1+np.exp(-(np.matmul(patients4, thetas2[:, None])))))  #prob of Y=1 # (sizex3) x (3x1) = (size, 1)
    
    #####################OLD part#########################
    #Xa = patients1[:, 2] # shape: size
    #g2 = ((Xa) + 0.5*((Xa)+np.sqrt(1+(Xa)**2)))*(1-rho1**2) + ((Xa) - 0.5*((Xa)+np.sqrt(1+(Xa)**2)))*(rho1**2)
    #Xa = g2 # size
    
    #calculate reward
    #get new coefficients given the covariate Xa has changed by running logit 
    #Y = np.random.binomial(1, 0.2, (self.size, 1))
    #patients2 = np.hstack([Y, np.reshape(patients1[:, 1], (self.size,1)), np.reshape(Xa, (self.size,1))]) #Y, Xs, Xa
    #run logit model to get coefficients, because their risk has changed (or use acitons to get risk using just new Xa??)
    #model2 = LogisticRegression().fit(patients2[:, 1:3], np.ravel(patients2[:, 0].astype(int)))
    #thetas2 = np.array([model2.intercept_[0], model2.coef_[0,0] , model2.coef_[0,1]]) #thetas2[0]: intercept; thetas2[1]: coef for Xs, thetas2[2] coef for Xa
    
    #patients3 = np.hstack([np.ones((self.size, 1)), patients2[:, 1:3]]) #1, Xs, Xa
    #rho3 = (1/(1+np.exp(-(np.matmul(patients3, thetas2[:, None])))))  #prob of Y=1 # (sizex3) x (3x1) = (size, 1)
    #rho3 = rho3.squeeze() # shape: size, individual risk
    #####################OLD part#########################
    
    #transform rho3 in a list to print individual risk (and not just mean risk of the hospitak's patient)
    rho3_list = rho3.tolist()
    self.mean_r = np.mean(rho3)
    
    #check if horizon is over, otherwise keep on going
    if self.horizon <= 0:
      done = True
    else:
      done = False
    #set the reward equal to the mean hospitalization rate
    reward = self.mean_r 
        
    self.state = self.patients[self.random_indices, :].reshape(2,) #not sure if with or without reshape
    
    #without action - simple logit on inital (non-intervened) dataset with Y, old Xa, Xs
    patients4 = np.hstack([Y, self.patients]) #shape (size, 3), 1st column of Y, 2nd columns Xs, 3rd column Xa
    model4 = LogisticRegression().fit(patients4[:, 1:3], np.ravel(patients4[:, 0].astype(int)))
    thetas4 = np.array([model4.intercept_[0], model4.coef_[0,0] , model4.coef_[0,1]]) #thetas4[0]: intercept; thetas4[1]: coef for Xs, thetas4[2] coef for Xa
    rho4 = (1/(1+np.exp(-(np.matmul(patients1, thetas4[:, None])))))  #prob of Y=1  #(sizex3) x (3x1) = (size, 1) #use patients1 because it's fine, it has self.patients
    rho4 = rho4.squeeze() # shape: size, individual risk
    rho4_list = rho4.tolist()
    reward4 = np.mean(rho4)
    
 
    
    #reduce the horizon
    self.horizon -= 1
    
    #set placeholder for infos
    info ={}    
    return self.state, reward,  reward4, rho3, rho4, thetas4, done, {}

#reset state and horizon    
  def reset(self):
    self.horizon = 2000
    
    #define dataset of patients with non-actionable covariate Xs and actionable covariate Xa
    self.patients = truncnorm.rvs(a=0, b= math.inf,size=(self.size,2)) #shape (size, 2), 1st columns is Xs, second is Xa
    
       
    self.random_indices = np.random.choice(self.size, size=1, replace=False)
    self.state = self.patients[self.random_indices, :].reshape(2,) 

    
    
    return self.state
