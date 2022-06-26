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

    #introduce some length
    self.horizon=200

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
    
    # bit modified 
    #-----------------------------------------------------------------------------------
    # e=0, t=0
    # observe patients (from reset function, self.patients) (Xs(0), Xa(0))
    
    #-----------------------------------------------------------------------------------
    #e=0, t=1    
    # observe same patients (Xs(1), Xa(1))=(Xs(0), Xa(0))
    # observe Y(1)
    Y_0(1) = np.random.binomial(1, 0.2, (self.size, 1))
    pat_0(1) = np.hstack([Y_0(1), self.patients]) # Y, Xs, Xa
    
    # predict f_0 = E[Y_0|X_0] using a logistic
    model_0 = LogisticRegression().fit(pat_0(1)[:, 1:3], np.ravel(pat_0(1)[:, 0].astype(int)))
    thetas_0 = np.array([model_0.intercept_[0], model_0.coef_[0,0] , model_0.coef_[0,1]]) #thetas0[0]: intercept; thetas0[1]: coef for Xs, thetas0[2] coef for Xa
    pat_00 = np.hstack([np.ones((self.size, 1)), self.patients]) #1, Xs, Xa
    f_0 = (1/(1+np.exp(-(np.matmul(pat_00, thetas_0[:, None])))))  #prob of Y=1 # (sizex3) x (3x1) = (size, 1)
    f_0=f_0.squeeze() 
    f_0_mean =  np.mean(f_0) # take mean across individuals
    
    # decide on initial actions
    initial_actions = thetas_0
    
    #-----------------------------------------------------------------------------------
    # e=1, t=0
    # observe patients (from reset function, self.patients) (Xs_1(0), Xa_1(0))
    pat_1(0)= np.hstack([np.ones((self.size, 1)),truncnorm.rvs(a=0, b= math.inf,size=(self.size,2))]) #shape (size, 3), (1, Xs, Xa)
    
    # compute rho_0(Xs_1(0), Xa_1(0)) - we're using covariates at e1 and thetas at e0
    rho_0 = (1/(1+np.exp(-(np.matmul(pat_1(0), initial_actions[:, None])))))  #prob of Y=1. # (sizex3) x (3x1) = (size, 1)
    
    # decide an intervention, use rho_0, Xa_1(0) 
    Xa = pat_1(0)[:, 2] # shape: size
    g_1 = intervention(Xa, rho_0)
    
    #-----------------------------------------------------------------------------------
    # e=1, t=1
    #update Xa_1(0) to Xa_1(1) with intervention
    Xa = g2 # size
    
    # observe Y_1(1)
    Y_1(1) = np.random.binomial(1, 0.2, (self.size, 1))
    pat_1(1) = np.hstack([Y_1(1), pat_1(0)[:, 1:3]) #shape (size, 3), (Y, Xs, Xa)
    
    # predict f_1 = E[Y_1|X_1(1)] using a logistic
    model_1 = LogisticRegression().fit(pat_1(1)[:, 1:3], np.ravel(pat_1(1)[:, 0].astype(int)))
    thetas_1 = np.array([model_1.intercept_[0], model_1.coef_[0,0] , model_1.coef_[0,1]]) #thetas1[0]: intercept; thetas1[1]: coef for Xs, thetas1[2] coef for Xa
    
    # use actions (thetas) from environment (then it is useless the logistic in model_1 (?))  
    theta0, theta1, theta2 = action  # actions(thetas) from env                      
    f_1 = (1/(1+np.exp(-(np.matmul(pat_00, action[:, None])))))  #prob of Y=1 # (sizex3) x (3x1) = (size, 1)
    f_1=f_1.squeeze() 
    f_1_mean =  np.mean(f_1) # take mean across individuals    
                          
    #-----------------------------------------------------------------------------------
    # naive result. use pat_1(0), Y_1(1)
    pat_naive =  pat_1(1)
    model_naive = LogisticRegression().fit(pat_naive[:, 1:3], np.ravel(pat_naive[:, 0].astype(int)))                        
    thetas_naive = np.array([model_naive.intercept_[0], model_naive.coef_[0,0] , model_naive.coef_[0,1]]) #thetas_n[0]: intercept; thetas_n[1]: coef for Xs, thetas_n[2] coef for Xa
    pat_naive00 = pat_1(0)
    rho_naive = (1/(1+np.exp(-(np.matmul(pat_naive00, thetas_naive[:, None])))))  #prob of Y=1 # (sizex3) x (3x1) = (size, 1)  
                            
    #-----------------------------------------------------------------------------------                        
    # we return:
    # - actions
    # - f_1_mean, that is f_1 = E[Y_1|X_1(1)]                      
    # - thetas_naive 
    # - rho_naive, that is E[Y_1|X_1(1)]   #?                       
                            
                            
    #MODEL FITTING CYCLE      
    
    #e=0, t=0
    #see patients (from reset function) (Xa(0), Xs(0)), no intervention
    list1 = np.zeros((self.size,self.horizon+1))
    rho_mean=[]
    rho_meanl=[]
    theta_list=[]
    theta_list_l=[]

    #e=0, t=1    
    #see same patients (Xa(1), Xs(1))=(Xa(0), Xs(0)) because no intervention occurred but this only at e=0
    #see Y (not necessary I guess)   
    theta0, theta1, theta2 = action  


    #compute rho
    patients1= np.hstack([np.ones((self.size, 1)), self.patients]) #shape (50, 3), 1st column of 1's, 2nd columns Xs, 3rd column Xa
    rho1 = (1/(1+np.exp(-(np.matmul(patients1, action[:, None])))))  #prob of Y=1  # (sizex3) x (3x1) = (size, 1)
    rho1 = rho1.squeeze() # shape: size, individual risk
    rho1_mean =  np.mean(rho1)

    list1[:,0]=rho1
    rho_mean.append(rho1_mean)
    theta_list.append(action)
    
    for i in range(self.horizon):
      #e=1, t=0
      #see new patients (Xa(0), Xs(0))
      patients2 = truncnorm.rvs(a=0, b= math.inf,size=(self.size,2)) #shape (size, 2), 1st columns is Xs, second is Xa
      Xa = patients2[:, 1] # shape: size
      #apply intervention and use rho from previous episode. g(rho_{e-1}, Xa(0))
      g2 = ((Xa) + 0.5*((Xa)+np.sqrt(1+(Xa)**2)))*(1-list1[:,i]**2) + ((Xa) - 0.5*((Xa)+np.sqrt(1+(Xa)**2)))*(list1[:,i]**2)
    
      #e=1, t=1
      #update Xa(0) to Xa(1) with intervention. Xa(1)=g(rho_{e-1}, Xa(0))
      Xa = g2 # size
      Y = np.random.binomial(1, 0.2, (self.size, 1))
      patients3 = np.hstack([Y, np.reshape(patients2[:, 0], (self.size,1)), np.reshape(Xa, (self.size,1))]) #Y, Xs, Xa
      model3 = LogisticRegression().fit(patients3[:, 1:3], np.ravel(patients3[:, 0].astype(int)))
      #compute rho by fitting model
      thetas3 = np.array([model3.intercept_[0], model3.coef_[0,0] , model3.coef_[0,1]]) #thetas2[0]: intercept; thetas2[1]: coef for Xs, thetas2[2] coef for Xa
      patients4 = np.hstack([np.ones((self.size, 1)), patients3[:, 1:3]]) #1, Xs, Xa
      rho4 = (1/(1+np.exp(-(np.matmul(patients4, thetas3[:, None])))))  #prob of Y=1 # (sizex3) x (3x1) = (size, 1)
      rho4=rho4.squeeze() 
      rho4_mean =  np.mean(rho4)
  
      list1[:, i+1]=rho4
      rho_mean.append(rho4_mean)  # mean reward in a cycle
      theta_list.append(thetas3) # list of all thetas assigned in a cycle
      #HERE. should repeat only dynamics of episode 1    
      
    #check if horizon is over, otherwise keep on going
    if rho_mean[-1] >= 0.2:
      done = True
    else:
      done = False
    #set the reward equal to the mean hospitalization rate
    reward = np.mean(rho_mean)
        
    self.state = self.patients[self.random_indices, :].reshape(2,) #not sure if with or without reshape
    
    
    #without action - simple logit on inital (non-intervened) dataset with Y, old Xa, Xs
    patients5 = np.hstack([Y, self.patients]) #shape (50, 3), 1st column of Y, 2nd columns Xs, 3rd column Xa
    model5 = LogisticRegression(fit_intercept=False).fit(patients5[:, 1:3], np.ravel(patients5[:, 0].astype(int)))
    thetas5 = np.array([model5.intercept_[0], model5.coef_[0,0] , model5.coef_[0,1]]) #thetas5[0]: coef for Xs, thetas4[1] coef for Xa
    rho5 = (1/(1+np.exp(-(np.matmul(patients1, thetas5[:, None])))))  #prob of Y=1  #(sizex3) x (3x1) = (size, 1) #use patients1 because it's fine, it has self.patients
    rho5 = rho5.squeeze() # shape: size, individual risk
    rho5_list = rho5.tolist()
    reward5 = np.mean(rho5)
    
    
    #set placeholder for infos
    info ={}    
    return self.state, reward, theta_list[0], theta_list[-1], reward5, done, {}
    # return state, reward (mean of population reward across all iterations),
    # first action assigned, last action assigned, naive reward
    
#reset state and horizon    
  def reset(self):
    self.horizon = 200
    
    #define dataset of patients with non-actionable covariate Xs and actionable covariate Xa
    self.patients = truncnorm.rvs(a=0, b= math.inf,size=(self.size,2)) #shape (size, 2), 1st columns is Xs, second is Xa
    
       
    self.random_indices = np.random.choice(self.size, size=1, replace=False)
    self.state = self.patients[self.random_indices, :].reshape(2,) 

    
    
    return self.state
