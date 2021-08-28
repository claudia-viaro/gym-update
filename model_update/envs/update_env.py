#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#packages environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
#from gym.utils import seeding
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
import pandas.testing as tm
import math
from sklearn.linear_model import LogisticRegression

#Gym environment - continuous

class UpdateEnv(gym.Env):
  def __init__(self):
    self.size = 50
    #get initial values for theta's
    #fit logit model to data
    self.df = pd.DataFrame(dict(
            Xs=np.random.normal(0,1,size=self.size),
            Xa=np.random.normal(0,1,size=self.size),
            Y=np.random.binomial(1, 0.5, self.size)))
    self.model = LogisticRegression().fit(self.df[["Xs", "Xa"]], np.ravel(self.df[["Y"]]))

    #extract theta parameters from the fitted logistic
    self.thetas = np.array([self.model.coef_[0,0] , self.model.coef_[0,1], self.model.intercept_[0]]) #thetas[1] coef for Xs, thetas[2] coef for Xa

    #set range for obs space
    #? not all values should be equaly likely to be sampled, this is missing here
    #? can I restrict the sampling space when an episode is run?
    self.minXa1 = pd.to_numeric(min(self.df[["Xa"]].values.flatten()))
    self.minXs1 = pd.to_numeric(min(self.df[["Xs"]].values.flatten()))
    
    self.maxXa1 = pd.to_numeric(max(self.df[["Xa"]].values.flatten()))
    self.maxXs1 = pd.to_numeric(max(self.df[["Xs"]].values.flatten()))
    
    self.min_Xas=np.array([np.float32(self.minXa1), np.float32(self.minXs1)])
    self.max_Xas=np.array([np.float32(self.maxXa1), np.float32(self.maxXs1)])


    #set range for action space
    self.high_th = np.array([2, 2, 2])
   
    #set ACTION SPACE
    #space.box handles continuous action space
    #it needs values for the bounds and the shape: low, high, shape (as in a tensor)
    #the bounds for a logit transformation of X's are 0, 1 (or min and max of the logit transform with initial values for theta)
    self.action_space = spaces.Box(
            low = np.float32(-self.high_th),
            high = np.float32(self.high_th))
    
    #set OBSERVATION SPACE
    #it is made of values for Xa, Xs for each observation
    self.observation_space = spaces.Box(low=self.min_Xas, 
                                   high=self.max_Xas, 
                                   dtype=np.float32)
    
    #set an initial state
    self.state=None 

    #introduce some length
    self.horizon=200
 

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]    

#take an action with the environment
#it returns the next observation, the immediate reward, whether the episode is over (done) and additional information    
#"action" argument is one value in the range of the action space (logit transform)
  def step(self, action):

    theta0, theta1, theta2 = action    
    data = []
    for self.state in self.dfa.itertuples(index=False):
      Xs = self.state[0]
      Xa = self.state[1]
      Y = self.state[2]
      Xsa=(theta0)+(theta1)*(Xs)+(theta2)*(Xa)
      rho2 = (1/(1+np.exp(-Xsa)))  #prob of Y=1
      g2 = ((Xa) + 0.5*((Xa)+np.sqrt(1+(Xa)**2)))*(1-rho2**2) + ((Xa) - 0.5*((Xa)+np.sqrt(1+(Xa)**2)))*(rho2**2)
      if (rho2>0.2):
        Xa=g2
      else: Xa=Xa 

      data.append([Xs, Xa, Y])
      
    df_new = pd.DataFrame(data, columns=['Xs', 'Xa', 'Y']) 

    model1 = LogisticRegression().fit(df_new[["Xs", "Xa"]], np.ravel(df_new[["Y"]].astype(int)))

    #extract theta parameters from the fitted logistic
    thetas1 = np.array([model1.coef_[0,0] , model1.coef_[0,1], model1.intercept_[0]])
    #extract theta parameter for Xa from the fitted logistic
    #theta_updated = np.array([model1.coef_[0,1]]) #thetas[1] coef for Xs, thetas[2] coef for Xa
    list1= []
    for i in df_new.itertuples(index=False):
      Xss = i[0] #no change
      Xaa = i[1] #no change
      YY = i[2] #no change

      Z1 = ((thetas1[0])+(thetas1[1])*(Xss)+(thetas1[2])*(Xaa))
      Prob1 = np.exp(Z1)/(1+np.exp(Z1)) #P(Y=1)
      list1.append([Xss, Xaa, YY, Prob1])

    list_new = pd.DataFrame(list1, columns=['Xs', 'Xa', 'Y', 'Prob1']) 
    self.Ynew_cumul = np.mean(list_new[["Prob1"]])
    
    #Z1 = ((thetas1[0])+(thetas1[1])*(df_new["Xs"])+(thetas1[2])*(df_new["Xa"])).values
    #Prob1 = (np.exp(Z1)).astype("float")/(1+(np.exp(Z1)).astype("float")) #P(Y=1)
    #self.Ynew_cumul = np.mean(Prob1)
    
    #check if horizon is over, otherwise keep on going
    if self.horizon <= 0:
      done = True
    else:
      done = False
    #depending on the value of self.state, apply a reward
    reward = self.Ynew_cumul 
        
    self.state = self.dfa.sample(n=1, random_state=1).values.reshape(3,)[0:2] 
    
    #reduce the horizon
    self.horizon -= 1
    
    #set placeholder for infos
    info ={}    
    return self.state, reward, thetas1, done, {}

#reset state and horizon    
  def reset(self):
    self.horizon = 200
    #done = False
    
    self.dfa = pd.DataFrame(dict(
            Xsa=np.random.normal(0,1,size=self.size),
            Xaa=np.random.normal(0,1,size=self.size),
            Ya=np.random.binomial(1, 0.5, self.size)))
       
    #fit logistic model
    self.modela = LogisticRegression().fit(self.dfa[["Xsa", "Xaa"]], np.ravel(self.dfa[["Ya"]]))
    
    #extract theta parameters from the fitted logistic
    self.thetasa = np.array([self.modela.coef_[0,0] , self.modela.coef_[0,1], self.modela.intercept_[0]]) #thetas[1] coef for Xs, thetas[2] coef for Xa
    
    self.state=self.dfa.sample(n=1, random_state=1).values.reshape(3,)[0:2]
    
    return self.state
